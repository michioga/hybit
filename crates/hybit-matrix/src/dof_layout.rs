//! Variable-field degrees of freedom (H1). Metadata only: never changes a solver's
//! mathematical matrix class or silently enables contact/nonlinear behavior.
//!
//! Blocks are ordered in the same order as the caller's *physical* DOF vector.
//! Permutation into solver-internal ordering is a separate, future facility.

use std::collections::BTreeSet;
use std::ops::Range;

use hybit_core::HybitError;

use crate::DofMask;

/// Stable entity-kind codes shared with `include/hybit.h` / `fortran/hybit.f90`.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum DofEntityKind {
    Node = 1,
    Edge = 2,
    Face = 3,
    Element = 4,
    Multiplier = 5,
    Constraint = 6,
    Other = 255,
}

impl DofEntityKind {
    pub fn from_code(code: u32) -> Result<Self, HybitError> {
        match code {
            1 => Ok(Self::Node),
            2 => Ok(Self::Edge),
            3 => Ok(Self::Face),
            4 => Ok(Self::Element),
            5 => Ok(Self::Multiplier),
            6 => Ok(Self::Constraint),
            255 => Ok(Self::Other),
            _ => Err(HybitError::InvalidArgument("unknown DOF entity kind")),
        }
    }
}

/// One contiguous block of components belonging to an entity and a field.
/// `field_id` is caller-defined; it is deliberately not restricted to
/// temperature/displacement/rotation or to 3/6-component FEM nodes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DofBlock {
    pub entity_kind: DofEntityKind,
    pub entity_id: u64,
    pub field_id: u32,
    pub components: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DofAddress {
    pub block_index: usize,
    pub component: u32,
}

/// Reusable description of variable-sized, mixed-field physical DOFs.
///
/// Each `(entity_kind, entity_id, field_id)` occurs at most once. The same
/// entity may have multiple fields (e.g. translation, rotation, temperature).
/// No assumption about SPD, element type, assembly, or contact is made.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FieldAwareDofLayout {
    blocks: Vec<DofBlock>,
    offsets: Vec<u64>,
}

impl FieldAwareDofLayout {
    pub fn new(blocks: Vec<DofBlock>) -> Result<Self, HybitError> {
        if blocks.is_empty() {
            return Err(HybitError::InvalidArgument(
                "DOF layout requires at least one block",
            ));
        }
        let mut seen = BTreeSet::new();
        let mut offsets = Vec::with_capacity(blocks.len() + 1);
        offsets.push(0_u64);
        let mut next = 0_u64;
        for block in &blocks {
            if block.components == 0 {
                return Err(HybitError::InvalidArgument(
                    "DOF block component count must be positive",
                ));
            }
            if !seen.insert((block.entity_kind, block.entity_id, block.field_id)) {
                return Err(HybitError::InvalidArgument(
                    "duplicate DOF entity/field block",
                ));
            }
            next = next
                .checked_add(u64::from(block.components))
                .ok_or(HybitError::SizeOverflow)?;
            offsets.push(next);
        }
        Ok(Self { blocks, offsets })
    }

    pub fn total_dofs(&self) -> u64 {
        *self.offsets.last().expect("nonempty layout offsets")
    }

    pub fn blocks(&self) -> &[DofBlock] {
        &self.blocks
    }

    pub fn block_range(&self, index: usize) -> Option<Range<u64>> {
        (index < self.blocks.len()).then(|| self.offsets[index]..self.offsets[index + 1])
    }

    pub fn locate(&self, dof: u64) -> Option<DofAddress> {
        if dof >= self.total_dofs() {
            return None;
        }
        let block_index = self.offsets.partition_point(|&offset| offset <= dof) - 1;
        Some(DofAddress {
            block_index,
            component: (dof - self.offsets[block_index]) as u32,
        })
    }

    /// Rejects a metadata/matrix mismatch without changing matrix contents.
    pub fn validate_square_matrix_size(
        &self,
        nrows: usize,
        ncols: usize,
    ) -> Result<(), HybitError> {
        let n = usize::try_from(self.total_dofs()).map_err(|_| HybitError::SizeOverflow)?;
        if nrows != n || ncols != n {
            return Err(HybitError::InvalidMatrix(
                "square matrix dimensions disagree with DOF layout",
            ));
        }
        Ok(())
    }

    /// Construct a packed field mask by writing entire 64-bit words at a time.
    /// Symbolic metadata generation; no numeric zero approximation is involved.
    pub fn field_mask(&self, field_id: u32) -> Result<DofMask, HybitError> {
        let n = usize::try_from(self.total_dofs()).map_err(|_| HybitError::SizeOverflow)?;
        let mut mask = DofMask::new(n);
        for (index, block) in self.blocks.iter().enumerate() {
            if block.field_id != field_id {
                continue;
            }
            let start =
                usize::try_from(self.offsets[index]).map_err(|_| HybitError::SizeOverflow)?;
            let end =
                usize::try_from(self.offsets[index + 1]).map_err(|_| HybitError::SizeOverflow)?;
            for word in start / 64..=(end - 1) / 64 {
                let first = if word == start / 64 { start % 64 } else { 0 };
                let last_exclusive = if word == (end - 1) / 64 {
                    (end - 1) % 64 + 1
                } else {
                    64
                };
                let high = if last_exclusive == 64 {
                    u64::MAX
                } else {
                    (1_u64 << last_exclusive) - 1
                };
                mask.or_word(word, high & (u64::MAX << first));
            }
        }
        Ok(mask)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block(kind: DofEntityKind, entity: u64, field: u32, n: u32) -> DofBlock {
        DofBlock {
            entity_kind: kind,
            entity_id: entity,
            field_id: field,
            components: n,
        }
    }

    #[test]
    fn mixed_thermal_beam_shell_solid_and_multiplier() {
        use DofEntityKind::*;
        let layout = FieldAwareDofLayout::new(vec![
            block(Node, 10, 1, 3),      // solid displacement
            block(Node, 11, 1, 3),      // beam translations
            block(Node, 11, 2, 3),      // beam rotations
            block(Node, 12, 1, 3),      // shell translations
            block(Node, 12, 2, 2),      // 5-DOF shell rotations
            block(Node, 10, 3, 1),      // temperature on same node
            block(Multiplier, 8, 4, 3), // contact normal/tangent multipliers
        ])
        .unwrap();
        assert_eq!(layout.total_dofs(), 18);
        assert_eq!(layout.block_range(4), Some(12..14));
        assert_eq!(
            layout.locate(12),
            Some(DofAddress {
                block_index: 4,
                component: 0
            })
        );
        assert_eq!(
            layout.locate(13),
            Some(DofAddress {
                block_index: 4,
                component: 1
            })
        );
        assert_eq!(
            layout.locate(14),
            Some(DofAddress {
                block_index: 5,
                component: 0
            })
        );
        assert_eq!(
            layout.locate(17),
            Some(DofAddress {
                block_index: 6,
                component: 2
            })
        );
        assert_eq!(layout.locate(18), None);
        assert_eq!(
            layout.field_mask(2).unwrap().indices(),
            vec![6, 7, 8, 12, 13]
        );
        assert_eq!(layout.field_mask(99).unwrap().count_ones(), 0);
        assert!(layout.validate_square_matrix_size(18, 18).is_ok());
        assert!(layout.validate_square_matrix_size(18, 17).is_err());
    }

    #[test]
    fn field_mask_across_bitmap_word_boundaries() {
        let layout = FieldAwareDofLayout::new(vec![
            block(DofEntityKind::Node, 1, 42, 63),
            block(DofEntityKind::Node, 2, 17, 3),
            block(DofEntityKind::Node, 3, 42, 70),
        ])
        .unwrap();
        let bits = layout.field_mask(42).unwrap();
        assert_eq!(bits.len(), 136);
        assert_eq!(bits.count_ones(), 133);
        assert!(!bits.contains(63));
        assert!(!bits.contains(64));
        assert!(!bits.contains(65));
        assert!(bits.contains(66));
        assert!(bits.contains(135));
        assert_eq!(bits.words().len(), 3);
    }

    #[test]
    fn rejects_ambiguous_or_empty_blocks() {
        use DofEntityKind::Node;
        assert!(FieldAwareDofLayout::new(vec![]).is_err());
        assert!(FieldAwareDofLayout::new(vec![block(Node, 1, 0, 0)]).is_err());
        assert!(
            FieldAwareDofLayout::new(vec![block(Node, 1, 3, 1), block(Node, 1, 3, 4)]).is_err()
        );
        assert!(FieldAwareDofLayout::new(vec![block(Node, 1, 3, 1), block(Node, 1, 4, 4)]).is_ok());
        assert!(DofEntityKind::from_code(7).is_err());
        assert_eq!(DofEntityKind::from_code(255).unwrap(), DofEntityKind::Other);
    }
}
