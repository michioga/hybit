mod abtm;
mod csr32;
mod dense_block;
mod dual_topology;
mod mask;
mod matrix_market;
mod metadata_first;
mod profile;
mod region;
mod topology;

pub use abtm::{AbtmConfig, AbtmMatrix, AbtmStats, TileDesc, TileKind, TILE_WIDTH};
pub use csr32::{Csr32Matrix, ParallelCsr32Operator};
pub use dense_block::{DenseBlockCsrOperator, DenseBlockSize};
pub use dual_topology::{AbtmDualTopology, AbtmDualTopologyStats};
pub use mask::DofMask;
pub use matrix_market::{
    read_matrix_market, read_matrix_market_from_reader, write_matrix_market_general,
    MatrixMarketError, MatrixMarketInfo, MatrixMarketSymmetry,
};
pub use metadata_first::{
    AbtmMetadataFirstMatrix, AbtmMetadataFirstStats, AbtmProductPruningStats,
};
pub use profile::{analyze_csr32, MatrixProfile};
pub use region::{
    extract_local_submatrix_pattern, grow_undirected_region, prepare_local_numeric_plan,
    region_multiplicity, AbtmLocalNumericPlan, AbtmLocalNumericPlanStats,
    AbtmLocalSubmatrixPattern, AbtmLocalSubmatrixPatternStats, AbtmRegionGrowth,
    AbtmRegionGrowthStats, AbtmRegionMultiplicity, AbtmRegionMultiplicityStats,
};
pub use topology::{
    AbtmTopology, AbtmTopologyRow, AbtmTopologyStats, AbtmTopologyWord, ABTM_TOPOLOGY_WORD_BITS,
};
