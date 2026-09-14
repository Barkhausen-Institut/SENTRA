# KVS API and JWT Identity Verification Specification

This document specifies the exact API calls for reading/writing versioned shares in the TDX-protected KVS and the JWT-bound identity verification mechanism.

## Current Implementation Status

### ✅ Versioned KVS API (Implemented)

The current implementation provides the following API calls:

#### **Write Operations**

**1. Single Node Write:**
```python
# Method: KVSNode.write(key, data, version)
# Location: ml_training/kvs.py, lines 34-57

def write(self, key: str, data: Any, version: int) -> bool:
    """
    Write data with version to a single KVS node
    Only accepts writes with version > current version (rollback protection)
    
    Args:
        key: Key to write
        data: Data to store (typically Share objects or lists of Shares)
        version: Version number (must be > current version)
    
    Returns:
        True if write succeeded, False if version not fresh (replay protection)
    
    Example:
        node = KVSNode(node_id=1)
        success = node.write("sample_0_node_1", share_list, version=1)
    """
```

**2. Quorum Write:**
```python
# Method: KVSCluster.write_with_quorum(key, data, version, quorum_size)
# Location: ml_training/kvs.py, lines 92-112

def write_with_quorum(self, key: str, data: Any, version: int, 
                     quorum_size: Optional[int] = None) -> bool:
    """
    Write to cluster with quorum consensus
    Requires majority (or specified quorum) of nodes to acknowledge
    
    Args:
        key: Key to write
        data: Data to write
        version: Version number
        quorum_size: Required quorum size (default: majority = n_nodes//2 + 1)
    
    Returns:
        True if quorum achieved, False otherwise
    
    Example:
        cluster = KVSCluster(node_ids=[1, 2, 3, 4, 5])
        success = cluster.write_with_quorum(
            "weights_v2", 
            weight_shares, 
            version=2,
            quorum_size=3  # Optional: specify quorum
        )
    """
```

#### **Read Operations**

**1. Single Node Read:**
```python
# Method: KVSNode.read(key, min_version)
# Location: ml_training/kvs.py, lines 59-75

def read(self, key: str, min_version: int = 0) -> Optional[KVSValue]:
    """
    Read data with minimum version requirement
    Ensures only fresh shares are returned (prevents stale data)
    
    Args:
        key: Key to read
        min_version: Minimum version required (default: 0 = any version)
    
    Returns:
        KVSValue if found and version >= min_version, None otherwise
        KVSValue contains: data, version, timestamp
    
    Example:
        node = KVSNode(node_id=1)
        value = node.read("sample_0_node_1", min_version=1)
        if value:
            shares = value.data  # List of Share objects
            version = value.version  # Actual version
    """
```

**2. Cluster Read with Min Version:**
```python
# Method: KVSCluster.read_with_min_version(key, min_version)
# Location: ml_training/kvs.py, lines 114-128

def read_with_min_version(self, key: str, min_version: int = 0) -> Optional[KVSValue]:
    """
    Read from cluster with minimum version requirement
    Returns value from first node that has version >= min_version
    
    Args:
        key: Key to read
        min_version: Minimum version required (default: 0)
    
    Returns:
        KVSValue if found, None otherwise
    
    Example:
        cluster = KVSCluster(node_ids=[1, 2, 3, 4, 5])
        value = cluster.read_with_min_version("weights_v2", min_version=2)
        if value:
            weights = value.data  # Weight shares
            assert value.version >= 2  # Freshness guaranteed
    """
```

**3. Get Latest Version:**
```python
# Method: KVSCluster.get_latest_version(key)
# Location: ml_training/kvs.py, lines 130-142

def get_latest_version(self, key: str) -> int:
    """
    Get latest version for a key across all nodes
    
    Args:
        key: Key to check
    
    Returns:
        Latest version number, or 0 if key doesn't exist
    
    Example:
        latest_v = cluster.get_latest_version("weights_v2")
        # Use latest_v + 1 for next write
    """
```

### **Usage Examples in Training Pipeline**

**Dataset Ingestion (Write):**
```python
# Location: ml_training/coordinator.py, lines 156-160

# Write sample shares with version v_D
self.kvs_cluster.write_with_quorum(
    sample_key,           # e.g., "sample_0_node_1"
    node_sample_shares,   # List[Share] - one per feature
    self.v_D              # Dataset version (monotonically increasing)
)

# Write label shares with same version
self.kvs_cluster.write_with_quorum(
    label_key,            # e.g., "label_0_node_1"
    node_label_share,     # Single Share
    self.v_D
)
```

**Mini-Batch Selection (Read):**
```python
# Location: ml_training/training_pipeline.py, lines 118-119

# Read with min_version to ensure fresh shares
sample_value = self.kvs_cluster.read_with_min_version(
    sample_key,  # e.g., "sample_0_node_1"
    v_D          # Minimum dataset version required
)

label_value = self.kvs_cluster.read_with_min_version(
    label_key,   # e.g., "label_0_node_1"
    v_D          # Minimum dataset version required
)
```

**Model Weight Updates (Write):**
```python
# Location: ml_training/coordinator.py, lines 329-331

# Increment model version
self.v_theta += 1

# Write updated weights with incremented version
success = self.kvs_cluster.write_with_quorum(
    f"weights_v{self.v_theta}",
    updated_weights,      # List[List[List[Share]]] - weight matrices
    self.v_theta          # New model version
)
```

---

## ❌ JWT-Bound Identity Verification (Not Yet Implemented)

### **Current Status**

The current implementation **does not include JWT authentication**. The following components are missing:

1. **JWT Token Generation** - No token issuance mechanism
2. **JWT Token Verification** - No signature verification
3. **Identity Binding** - No binding between JWT claims and TDX enclave identity
4. **Access Control** - No authorization based on JWT claims

### **Required Implementation**

#### **1. JWT Token Structure**

For TDX-protected KVS, JWT tokens should include:

```json
{
  "header": {
    "alg": "RS256",
    "typ": "JWT"
  },
  "payload": {
    "sub": "node_1",                    // Subject (node ID)
    "iss": "sentra-auth-service",       // Issuer
    "aud": "sentra-kvs",                // Audience
    "exp": 1234567890,                  // Expiration
    "iat": 1234567800,                  // Issued at
    "tdx_measurement": "0xabc123...",   // TDX MRENCLAVE measurement
    "tdx_config_svn": 123,              // TDX CONFIG SVN
    "node_id": 1,                       // Node identifier
    "permissions": ["read", "write"]    // Access permissions
  },
  "signature": "RS256_signature..."
}
```

#### **2. Required API Extensions**

**Enhanced Write with JWT:**
```python
# Proposed: KVSNode.write_with_auth(key, data, version, jwt_token)
def write_with_auth(self, key: str, data: Any, version: int, 
                   jwt_token: str) -> bool:
    """
    Write data with version and JWT authentication
    
    Args:
        key: Key to write
        data: Data to store
        version: Version number
        jwt_token: JWT token for authentication
    
    Returns:
        True if write succeeded and JWT verified, False otherwise
    
    Steps:
        1. Verify JWT signature using public key
        2. Check JWT expiration (exp claim)
        3. Verify TDX measurement matches enclave
        4. Check write permission in JWT claims
        5. Verify node_id matches JWT subject
        6. Perform version check (version > current)
        7. Write data if all checks pass
    """
    # Verify JWT
    claims = verify_jwt_token(jwt_token)
    if not claims:
        return False
    
    # Check TDX measurement binding
    if not verify_tdx_measurement(claims['tdx_measurement']):
        return False
    
    # Check permissions
    if 'write' not in claims.get('permissions', []):
        return False
    
    # Verify node_id matches
    if claims['sub'] != f"node_{self.node_id}":
        return False
    
    # Perform versioned write
    return self.write(key, data, version)
```

**Enhanced Read with JWT:**
```python
# Proposed: KVSNode.read_with_auth(key, min_version, jwt_token)
def read_with_auth(self, key: str, min_version: int, 
                  jwt_token: str) -> Optional[KVSValue]:
    """
    Read data with minimum version and JWT authentication
    
    Args:
        key: Key to read
        min_version: Minimum version required
        jwt_token: JWT token for authentication
    
    Returns:
        KVSValue if found, verified, and authorized, None otherwise
    
    Steps:
        1. Verify JWT signature
        2. Check JWT expiration
        3. Verify TDX measurement
        4. Check read permission
        5. Verify node_id matches
        6. Perform versioned read
    """
    # Verify JWT
    claims = verify_jwt_token(jwt_token)
    if not claims:
        return None
    
    # Check TDX measurement binding
    if not verify_tdx_measurement(claims['tdx_measurement']):
        return None
    
    # Check permissions
    if 'read' not in claims.get('permissions', []):
        return None
    
    # Verify node_id matches
    if claims['sub'] != f"node_{self.node_id}":
        return None
    
    # Perform versioned read
    return self.read(key, min_version)
```

#### **3. JWT Verification Implementation**

**Required Module: `ml_training/jwt_auth.py`**

```python
import jwt
from typing import Dict, Optional
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
import tdx_attestation  # TDX measurement verification

class JWTAuthVerifier:
    """Verifies JWT tokens bound to TDX enclave identities"""
    
    def __init__(self, public_key_path: str):
        """
        Initialize JWT verifier
        
        Args:
            public_key_path: Path to public key for signature verification
        """
        with open(public_key_path, 'rb') as f:
            self.public_key = serialization.load_pem_public_key(f.read())
    
    def verify_jwt_token(self, token: str) -> Optional[Dict]:
        """
        Verify JWT token signature and extract claims
        
        Args:
            token: JWT token string
        
        Returns:
            Claims dictionary if valid, None otherwise
        """
        try:
            # Verify signature and decode
            claims = jwt.decode(
                token,
                self.public_key,
                algorithms=['RS256'],
                audience='sentra-kvs',
                issuer='sentra-auth-service'
            )
            return claims
        except jwt.ExpiredSignatureError:
            return None  # Token expired
        except jwt.InvalidSignatureError:
            return None  # Invalid signature
        except jwt.InvalidTokenError:
            return None  # Invalid token
    
    def verify_tdx_measurement(self, claimed_measurement: str) -> bool:
        """
        Verify that TDX measurement matches enclave
        
        Args:
            claimed_measurement: MRENCLAVE from JWT claims
        
        Returns:
            True if measurement matches, False otherwise
        """
        # Get actual TDX measurement from enclave
        actual_measurement = tdx_attestation.get_mrenclave()
        
        # Compare measurements
        return claimed_measurement == actual_measurement
    
    def verify_node_identity(self, claims: Dict, node_id: int) -> bool:
        """
        Verify node identity matches JWT subject
        
        Args:
            claims: JWT claims dictionary
            node_id: Expected node ID
        
        Returns:
            True if identity matches, False otherwise
        """
        expected_subject = f"node_{node_id}"
        return claims.get('sub') == expected_subject
```

#### **4. TDX Measurement Integration**

**Required: TDX Attestation Module**

```python
# ml_training/tdx_attestation.py

def get_mrenclave() -> str:
    """
    Get TDX MRENCLAVE measurement from current enclave
    
    Returns:
        Hex string of MRENCLAVE measurement
    """
    # In TDX enclave, use TDX attestation APIs
    # This would call TDX-specific functions to get measurement
    # For now, placeholder:
    import tdx_quote
    quote = tdx_quote.get_quote()
    return quote.mrenclave.hex()

def verify_tdx_config_svn(claimed_svn: int) -> bool:
    """
    Verify TDX CONFIG SVN matches
    
    Args:
        claimed_svn: CONFIG SVN from JWT
    
    Returns:
        True if matches, False otherwise
    """
    actual_svn = get_tdx_config_svn()
    return claimed_svn == actual_svn
```

---

## **Complete API Specification**

### **TDX-Protected KVS API Calls**

#### **Write Operations**

```python
# Single node write (inside TDX enclave)
success = kvs_node.write_with_auth(
    key="sample_0_node_1",
    data=share_list,           # List[Share]
    version=1,                 # Monotonically increasing
    jwt_token=jwt_token        # JWT with TDX measurement
)

# Quorum write (coordinator)
success = kvs_cluster.write_with_quorum_auth(
    key="weights_v2",
    data=weight_shares,        # List[List[List[Share]]]
    version=2,
    jwt_token=jwt_token,
    quorum_size=3              # Optional
)
```

#### **Read Operations**

```python
# Single node read (inside TDX enclave)
value = kvs_node.read_with_auth(
    key="sample_0_node_1",
    min_version=1,             # Minimum version required
    jwt_token=jwt_token        # JWT with TDX measurement
)

# Cluster read (coordinator)
value = kvs_cluster.read_with_min_version_auth(
    key="weights_v2",
    min_version=2,
    jwt_token=jwt_token
)
```

### **JWT Verification Flow**

```
┌─────────────┐
│   Client    │
│  (Node 1)   │
└──────┬──────┘
       │
       │ 1. Request JWT token
       │    (with TDX measurement)
       ▼
┌─────────────────┐
│  Auth Service   │
│  (Issues JWT)   │
└──────┬──────────┘
       │
       │ 2. JWT token
       │    (signed, includes TDX measurement)
       ▼
┌─────────────┐
│   Client    │
│  (Node 1)   │
└──────┬──────┘
       │
       │ 3. Write/Read request
       │    (key, data, version, JWT)
       ▼
┌──────────────────┐
│  TDX KVS Node    │
│  (Inside Enclave)│
└──────┬───────────┘
       │
       │ 4. Verify JWT:
       │    - Signature (RS256)
       │    - Expiration
       │    - TDX measurement match
       │    - Node ID match
       │    - Permissions
       │
       │ 5. Perform operation:
       │    - Version check
       │    - Write/Read
       ▼
┌─────────────┐
│   Success   │
│   or Error  │
└─────────────┘
```

---

## **Implementation Checklist**

### **Current Status:**
- ✅ Versioned read/write API (`read`, `write`, `read_with_min_version`, `write_with_quorum`)
- ✅ Rollback protection (version checking)
- ✅ Quorum consensus
- ❌ JWT token generation
- ❌ JWT signature verification
- ❌ TDX measurement binding
- ❌ Identity-based access control
- ❌ JWT expiration handling

### **Required Additions:**

1. **Create `ml_training/jwt_auth.py`**
   - JWT token verification
   - Signature validation
   - Claim extraction

2. **Create `ml_training/tdx_attestation.py`**
   - TDX measurement retrieval
   - CONFIG SVN verification
   - Enclave identity binding

3. **Enhance `ml_training/kvs.py`**
   - Add `write_with_auth()` method
   - Add `read_with_auth()` method
   - Add `write_with_quorum_auth()` method
   - Add `read_with_min_version_auth()` method

4. **Update `ml_training/coordinator.py`**
   - Pass JWT tokens to KVS operations
   - Handle JWT expiration/refresh

5. **Create Auth Service**
   - JWT token issuance
   - TDX measurement validation
   - Public key distribution

---

## **Security Considerations**

1. **JWT Token Lifetime**: Tokens should have short expiration (e.g., 1 hour)
2. **Key Rotation**: Public keys should be rotatable
3. **TDX Measurement Binding**: Critical - ensures code running in correct enclave
4. **Version Freshness**: `min_version` prevents replay attacks
5. **Quorum Requirements**: Prevents single-point-of-failure attacks
6. **Permission Granularity**: Fine-grained permissions (read/write per key prefix)

---

## **Example Complete Flow**

```python
# 1. Get JWT token (outside enclave, from auth service)
jwt_token = auth_service.get_token(
    node_id=1,
    tdx_measurement=get_tdx_measurement(),
    permissions=["read", "write"]
)

# 2. Inside TDX enclave - Write with JWT
kvs_node = KVSNode(node_id=1)
success = kvs_node.write_with_auth(
    key="sample_0_node_1",
    data=share_list,
    version=1,
    jwt_token=jwt_token
)

# 3. Inside TDX enclave - Read with JWT
value = kvs_node.read_with_auth(
    key="sample_0_node_1",
    min_version=1,
    jwt_token=jwt_token
)

if value:
    shares = value.data
    assert value.version >= 1  # Freshness guaranteed
```

---

## **Summary**

**Current API Calls:**
- `KVSNode.write(key, data, version)` ✅
- `KVSNode.read(key, min_version)` ✅
- `KVSCluster.write_with_quorum(key, data, version, quorum_size)` ✅
- `KVSCluster.read_with_min_version(key, min_version)` ✅

**Missing JWT Integration:**
- JWT token verification ❌
- TDX measurement binding ❌
- Identity-based access control ❌

**Next Steps:**
1. Implement JWT verification module
2. Add TDX attestation integration
3. Enhance KVS methods with JWT authentication
4. Create auth service for token issuance

