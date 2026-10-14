mod abtm;
mod csr32;
mod dense_block;
mod dof_layout;
mod dual_topology;
mod field_region;
mod mask;
mod matrix_market;
mod metadata_first;
mod prepared_field_region;
mod profile;
mod region;
mod restricted;
mod topology;

pub use abtm::{AbtmConfig, AbtmMatrix, AbtmStats, TileDesc, TileKind, TILE_WIDTH};
pub use csr32::{Csr32Matrix, ParallelCsr32Operator};
pub use dense_block::{DenseBlockCsrOperator, DenseBlockSize};
pub use dof_layout::{DofAddress, DofBlock, DofEntityKind, FieldAwareDofLayout};
pub use dual_topology::{AbtmDualTopology, AbtmDualTopologyStats};
pub use field_region::{
    close_to_field_blocks, field_block_owners, grow_field_aware_region, FieldAwareRegionGrowth,
    FieldAwareRegionStats,
};
pub use mask::DofMask;
pub use matrix_market::{
    read_matrix_market, read_matrix_market_from_reader, write_matrix_market_general,
    MatrixMarketError, MatrixMarketInfo, MatrixMarketSymmetry,
};
pub use metadata_first::{
    AbtmMetadataFirstMatrix, AbtmMetadataFirstStats, AbtmProductPruningStats,
};
pub use prepared_field_region::{PreparedFieldAwareLocalRegion, PreparedFieldAwareTopology};
pub use profile::{analyze_csr32, MatrixProfile};
pub use region::{
    extract_local_submatrix_pattern, grow_undirected_region, prepare_local_numeric_plan,
    region_multiplicity, AbtmLocalNumericPlan, AbtmLocalNumericPlanStats,
    AbtmLocalSubmatrixPattern, AbtmLocalSubmatrixPatternStats, AbtmRegionGrowth,
    AbtmRegionGrowthStats, AbtmRegionMultiplicity, AbtmRegionMultiplicityStats,
};
pub use restricted::{PreparedColumnRestrictedCsrOperator, PreparedLocalCsrOperator};
pub use topology::{
    AbtmTopology, AbtmTopologyRow, AbtmTopologyStats, AbtmTopologyWord, ABTM_TOPOLOGY_WORD_BITS,
};
