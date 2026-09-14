"""
SENTRA ML Training Pipeline
"""

from ml_training.secret_sharing import (
    Share,
    ShamirSecretSharing,
    PackedShamirSecretSharing
)
from ml_training.kvs import (
    KVSNode,
    KVSCluster,
    KVSValue
)
from ml_training.mpc_engine import PackedMPCEngine
from ml_training.coordinator import TrainingCoordinator, SafetyBoundChecker
from ml_training.training_pipeline import SentraTrainingPipeline
from ml_training.beaver_triples import (
    BeaverTriple,
    BeaverTripleGenerator,
    BeaverTriplePool,
    SecureMultiplier
)
from ml_training.secure_comparison import (
    SecureComparator,
    SecureClipper
)
from ml_training.secure_division import (
    SecureDivider,
    SecureAverager
)
from ml_training.secure_matrix_ops import (
    SecureMatrixMultiplier,
    SecureMatrixOperations,
    GPUMatrixAccelerator
)
from ml_training.secure_comm import (
    SecureChannel,
    SecureMPCNetwork,
    MessageType,
    create_mpc_network
)
from ml_training.reconstruction import (
    SecureReconstruction,
    MPCReconstructionManager,
    create_reconstruction_manager
)
try:
    from ml_training.sentra_training_node import (
        NodeIdentity,
        QuorumConfig,
        RetryConfig,
        TrainingConfig,
        VersionedValue,
        MiniBatchShares,
        VersionedKVSClient,
        MembershipTracker,
        SentraTrainingNode,
    )
    _HAS_SENTRA_NODE_RUNTIME = True
except Exception:
    # Optional runtime dependency path (coordination/kvstore/mpc) may be absent
    # for standard pipeline-only runs.
    _HAS_SENTRA_NODE_RUNTIME = False

__all__ = [
    'Share',
    'ShamirSecretSharing',
    'PackedShamirSecretSharing',
    'KVSNode',
    'KVSCluster',
    'KVSValue',
    'PackedMPCEngine',
    'TrainingCoordinator',
    'SafetyBoundChecker',
    'SentraTrainingPipeline',
    'BeaverTriple',
    'BeaverTripleGenerator',
    'BeaverTriplePool',
    'SecureMultiplier',
    'SecureComparator',
    'SecureClipper',
    'SecureDivider',
    'SecureAverager',
    'SecureMatrixMultiplier',
    'SecureMatrixOperations',
    'GPUMatrixAccelerator',
    'SecureChannel',
    'SecureMPCNetwork',
    'MessageType',
    'create_mpc_network',
    'SecureReconstruction',
    'MPCReconstructionManager',
    'create_reconstruction_manager',
]

if _HAS_SENTRA_NODE_RUNTIME:
    __all__.extend([
        'NodeIdentity',
        'QuorumConfig',
        'RetryConfig',
        'TrainingConfig',
        'VersionedValue',
        'MiniBatchShares',
        'VersionedKVSClient',
        'MembershipTracker',
        'SentraTrainingNode',
    ])
