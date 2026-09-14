# SENTRA Versioned KVS: Exact get() and put() Calls

This document shows the exact `get(key, min_version)` and `put(key, value, version)` calls used in SENTRA's versioned KVS and explains how version checks prevent rollback attacks.

## API Mapping

In SENTRA's implementation:
- **`put(key, value, version)`** → `KVSNode.write(key, data, version)`
- **`get(key, min_version)`** → `KVSNode.read(key, min_version)`

---

## Exact `put(key, value, version)` Implementation

### **Single Node Write (Base Operation)**

**Location:** `ml_training/kvs.py`, lines 34-57

```python
def write(self, key: str, data: Any, version: int) -> bool:
    """
    PUT operation: Write data with version
    Only accepts writes with version > current version (rollback protection)
    
    Args:
        key: Key to write
        data: Data to store (Share objects or lists of Shares)
        version: Version number (must be > current version)
    
    Returns:
        True if write succeeded, False if version check failed (rollback prevented)
    """
    import time
    
    # ═══════════════════════════════════════════════════════════════
    # ROLLBACK PREVENTION: Version Check
    # ═══════════════════════════════════════════════════════════════
    if key in self.store:
        current_version = self.store[key].version
        
        # CRITICAL: Only accept versions STRICTLY GREATER than current
        # This prevents:
        #   1. Replay attacks (same version)
        #   2. Rollback attacks (older version)
        if version <= current_version:
            return False  # Version not fresh - ROLLBACK PREVENTED
    
    # Version check passed - safe to write
    self.store[key] = KVSValue(
        data=data,
        version=version,
        timestamp=time.time()
    )
    return True
```

### **Quorum Write (Cluster Operation)**

**Location:** `ml_training/kvs.py`, lines 92-112

```python
def write_with_quorum(self, key: str, data: Any, version: int, 
                     quorum_size: Optional[int] = None) -> bool:
    """
    PUT with quorum: Write to cluster with quorum consensus
    Each node performs version check independently
    
    Args:
        key: Key to write
        data: Data to write
        version: Version number
        quorum_size: Required quorum size (default: majority)
    
    Returns:
        True if quorum achieved AND all nodes passed version check
    """
    if quorum_size is None:
        quorum_size = (self.n_nodes // 2) + 1
    
    # Write to all nodes - each performs version check
    successes = 0
    for node in self.nodes.values():
        # Each node.write() performs version check:
        #   - If version <= current_version: returns False
        #   - If version > current_version: returns True
        if node.write(key, data, version):
            successes += 1
    
    # Quorum achieved only if enough nodes passed version check
    return successes >= quorum_size
```

---

## Exact `get(key, min_version)` Implementation

### **Single Node Read (Base Operation)**

**Location:** `ml_training/kvs.py`, lines 59-75

```python
def read(self, key: str, min_version: int = 0) -> Optional[KVSValue]:
    """
    GET operation: Read data with minimum version requirement
    Ensures only fresh shares are returned (prevents stale data)
    
    Args:
        key: Key to read
        min_version: Minimum version required (default: 0 = any version)
    
    Returns:
        KVSValue if found and version >= min_version, None otherwise
        KVSValue contains: data, version, timestamp
    """
    if key not in self.store:
        return None
    
    value = self.store[key]
    
    # ═══════════════════════════════════════════════════════════════
    # FRESHNESS CHECK: Minimum Version Requirement
    # ═══════════════════════════════════════════════════════════════
    # Only return data if version is >= min_version
    # This prevents:
    #   1. Reading stale data (older versions)
    #   2. Using outdated shares in training
    if value.version < min_version:
        return None  # Version too old - STALE DATA REJECTED
    
    return value
```

### **Cluster Read with Min Version**

**Location:** `ml_training/kvs.py`, lines 114-128

```python
def read_with_min_version(self, key: str, min_version: int = 0) -> Optional[KVSValue]:
    """
    GET with min_version: Read from cluster with minimum version
    Returns value from first node that has version >= min_version
    
    Args:
        key: Key to read
        min_version: Minimum version required
    
    Returns:
        KVSValue if found with sufficient version, None otherwise
    """
    for node in self.nodes.values():
        # Each node.read() performs freshness check:
        #   - If version < min_version: returns None
        #   - If version >= min_version: returns KVSValue
        value = node.read(key, min_version)
        if value is not None:
            return value
    return None
```

---

## Actual Usage Examples from SENTRA Codebase

### **Example 1: Dataset Ingestion (PUT)**

**Location:** `ml_training/coordinator.py`, lines 156-161

```python
# PUT operation: Write sample shares with version v_D
self.kvs_cluster.write_with_quorum(
    sample_key,           # Key: "sample_0_node_1"
    node_sample_shares,   # Value: List[Share] - one per feature
    self.v_D              # Version: Dataset version (monotonically increasing)
)

# PUT operation: Write label shares with same version
self.kvs_cluster.write_with_quorum(
    label_key,            # Key: "label_0_node_1"
    node_label_share,     # Value: Single Share
    self.v_D              # Version: Same dataset version
)
```

**Flow:**
1. `write_with_quorum()` calls `node.write()` on each node
2. Each `node.write()` checks: `if version <= current_version: return False`
3. If all checks pass, data is written with new version
4. Returns `True` only if quorum of nodes accepted the write

### **Example 2: Mini-Batch Selection (GET)**

**Location:** `ml_training/training_pipeline.py`, lines 118-119

```python
# GET operation: Read sample shares with minimum version v_D
sample_value = self.kvs_cluster.read_with_min_version(
    sample_key,  # Key: "sample_0_node_1"
    v_D          # min_version: Minimum dataset version required
)

# GET operation: Read label shares with minimum version v_D
label_value = self.kvs_cluster.read_with_min_version(
    label_key,   # Key: "label_0_node_1"
    v_D          # min_version: Minimum dataset version required
)

# Check if data is fresh
if sample_value and label_value:
    # Both have version >= v_D - safe to use
    sample_feature_shares = sample_value.data
    label_share = label_value.data
    assert sample_value.version >= v_D  # Freshness guaranteed
    assert label_value.version >= v_D   # Freshness guaranteed
```

**Flow:**
1. `read_with_min_version()` calls `node.read(key, min_version)` on each node
2. Each `node.read()` checks: `if value.version < min_version: return None`
3. Returns first value with `version >= min_version`
4. If no node has fresh enough data, returns `None`

### **Example 3: Model Weight Updates (PUT)**

**Location:** `ml_training/coordinator.py`, lines 326-331

```python
# Increment model version (monotonically increasing)
self.v_theta += 1

# PUT operation: Write updated weights with incremented version
success = self.kvs_cluster.write_with_quorum(
    f"weights_v{self.v_theta}",  # Key: "weights_v2"
    updated_weights,              # Value: List[List[List[Share]]] - weight matrices
    self.v_theta                  # Version: New model version (incremented)
)

if not success:
    # Version check failed - possible rollback attempt
    raise RuntimeError("Failed to commit weights - version check failed")
```

**Flow:**
1. Version incremented: `v_theta = 2` (was 1)
2. `write_with_quorum()` attempts to write with version 2
3. Each node checks: `if 2 <= current_version: return False`
4. If current_version is 1, check passes: `2 > 1` ✓
5. If current_version is already 2, check fails: `2 <= 2` ✗ (prevents duplicate write)
6. If current_version is 3, check fails: `2 <= 3` ✗ (prevents rollback)

---

## How Version Check Prevents Rollback

### **Rollback Attack Scenario**

**Attack Goal:** Replace newer data with older data to revert system state

**Example Attack:**
```
Timeline:
  t1: PUT("key", "value_v2", version=2)  → Success
  t2: PUT("key", "value_v1", version=1)  → ATTACK: Try to rollback
```

### **Rollback Prevention Mechanism**

#### **Step 1: Version Check in PUT**

```python
# In KVSNode.write() - line 47-50
if key in self.store:
    current_version = self.store[key].version
    
    # CRITICAL CHECK: version must be STRICTLY GREATER
    if version <= current_version:
        return False  # ROLLBACK PREVENTED
```

**Attack Analysis:**
```
State after t1:
  store["key"] = KVSValue(data="value_v2", version=2)

Attack at t2:
  write("key", "value_v1", version=1)
  
  Check: if 1 <= 2:  # TRUE
    return False     # ROLLBACK PREVENTED ✓
```

#### **Step 2: Monotonic Version Enforcement**

**Version numbers must be monotonically increasing:**

```python
# In TrainingCoordinator.ingest_dataset() - line 128
self.v_D += 1  # Always increment before write

# In TrainingCoordinator.commit_weights() - line 326
self.v_theta += 1  # Always increment before write
```

**This ensures:**
- Version 1 → Version 2 → Version 3 → ...
- Never: Version 2 → Version 1 (rollback)
- Never: Version 2 → Version 2 (replay)

#### **Step 3: Freshness Check in GET**

```python
# In KVSNode.read() - line 72-73
if value.version < min_version:
    return None  # STALE DATA REJECTED
```

**This prevents:**
- Reading old data even if it exists in store
- Using outdated shares in training
- Stale data injection attacks

### **Complete Rollback Prevention Flow**

```
┌─────────────────────────────────────────────────────────────┐
│  PUT Operation: write(key, data, version)                  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────┐
        │  Key exists in store?             │
        └───────────┬───────────────────────┘
                    │
          ┌─────────┴─────────┐
          │                   │
         YES                  NO
          │                   │
          ▼                   ▼
┌─────────────────┐   ┌──────────────────┐
│ Get current_v   │   │ Write allowed    │
│ = store[key].v  │   │ (new key)        │
└────────┬────────┘   └────────┬─────────┘
         │                     │
         ▼                     │
┌─────────────────┐            │
│ version >       │            │
│ current_v?      │            │
└────────┬────────┘            │
         │                     │
    ┌────┴────┐                │
    │         │                │
   YES        NO               │
    │         │                │
    ▼         ▼                ▼
┌───────┐ ┌──────────┐   ┌──────────────┐
│ Write │ │ REJECT   │   │ Write        │
│ Allowed│ │ ROLLBACK │   │ Allowed      │
└───────┘ └──────────┘   └──────────────┘
```

### **Attack Vectors Prevented**

#### **1. Direct Rollback Attack**
```python
# Attacker tries to write older version
put("key", old_data, version=1)  # Current version is 2

# Prevention:
if 1 <= 2:  # TRUE
    return False  # ROLLBACK PREVENTED
```

#### **2. Replay Attack**
```python
# Attacker tries to replay same version
put("key", data, version=2)  # Current version is 2

# Prevention:
if 2 <= 2:  # TRUE
    return False  # REPLAY PREVENTED
```

#### **3. Stale Data Injection**
```python
# Attacker tries to read old data
get("key", min_version=3)  # Current version is 2

# Prevention:
if 2 < 3:  # TRUE
    return None  # STALE DATA REJECTED
```

#### **4. Version Number Manipulation**
```python
# Attacker tries to skip versions
put("key", data, version=5)  # Current version is 2

# Prevention:
if 5 > 2:  # TRUE
    # Write allowed, but this creates a gap
    # System should validate version continuity
    # (Additional check can be added)
```

---

## Version Check Logic: Step-by-Step

### **PUT Operation Version Check**

```python
def write(self, key: str, data: Any, version: int) -> bool:
    # Step 1: Check if key exists
    if key in self.store:
        # Step 2: Get current version
        current_version = self.store[key].version
        
        # Step 3: CRITICAL VERSION CHECK
        # Only allow versions STRICTLY GREATER than current
        if version <= current_version:
            # Step 4: REJECT - Rollback/Replay prevented
            return False
        
        # Step 5: Version check passed - safe to write
        # (version > current_version)
    
    # Step 6: Write new data with new version
    self.store[key] = KVSValue(
        data=data,
        version=version,  # New version stored
        timestamp=time.time()
    )
    return True
```

### **GET Operation Freshness Check**

```python
def read(self, key: str, min_version: int = 0) -> Optional[KVSValue]:
    # Step 1: Check if key exists
    if key not in self.store:
        return None
    
    # Step 2: Get stored value
    value = self.store[key]
    
    # Step 3: FRESHNESS CHECK
    # Only return data if version is >= min_version
    if value.version < min_version:
        # Step 4: REJECT - Stale data prevented
        return None
    
    # Step 5: Freshness check passed
    # (value.version >= min_version)
    return value
```

---

## Complete Example: Training Pipeline

### **Dataset Ingestion (PUT)**

```python
# Location: ml_training/coordinator.py:119-163

def ingest_dataset(self, dataset, labels):
    # Step 1: Increment version (monotonic)
    self.v_D += 1  # v_D: 0 → 1
    
    for i, (sample, label) in enumerate(zip(dataset, labels)):
        # Step 2: Create shares
        feature_shares = self.pss.shamir.share(feature_int, ...)
        
        # Step 3: PUT with version
        self.kvs_cluster.write_with_quorum(
            f"sample_{i}_node_1",
            feature_shares,
            self.v_D  # version = 1
        )
        
        # Version check in each node:
        #   - If key doesn't exist: Write allowed ✓
        #   - If key exists with version 0: 1 > 0 → Write allowed ✓
        #   - If key exists with version 1: 1 <= 1 → Write rejected ✗
        #   - If key exists with version 2: 1 <= 2 → Write rejected ✗
```

### **Mini-Batch Selection (GET)**

```python
# Location: ml_training/training_pipeline.py:118-119

# Step 1: GET with min_version
sample_value = self.kvs_cluster.read_with_min_version(
    "sample_0_node_1",
    v_D  # min_version = 1
)

# Freshness check in each node:
#   - If stored version is 0: 0 < 1 → Reject (stale) ✗
#   - If stored version is 1: 1 >= 1 → Accept ✓
#   - If stored version is 2: 2 >= 1 → Accept ✓

if sample_value:
    # Step 2: Use fresh data
    shares = sample_value.data
    assert sample_value.version >= v_D  # Guaranteed fresh
```

### **Model Update (PUT)**

```python
# Location: ml_training/coordinator.py:326-331

# Step 1: Increment version
self.v_theta += 1  # v_theta: 1 → 2

# Step 2: PUT with incremented version
success = self.kvs_cluster.write_with_quorum(
    f"weights_v{self.v_theta}",  # "weights_v2"
    updated_weights,
    self.v_theta  # version = 2
)

# Version check in each node:
#   - Current version is 1: 2 > 1 → Write allowed ✓
#   - Current version is 2: 2 <= 2 → Write rejected ✗ (duplicate)
#   - Current version is 3: 2 <= 3 → Write rejected ✗ (rollback)
```

---

## Summary

### **PUT Operation (`write(key, data, version)`)**

**Exact Call:**
```python
kvs_node.write(key="sample_0_node_1", 
               data=share_list, 
               version=1)
```

**Rollback Prevention:**
- Checks: `if version <= current_version: return False`
- Only allows: `version > current_version`
- Prevents: Rollback, replay, duplicate writes

### **GET Operation (`read(key, min_version)`)**

**Exact Call:**
```python
value = kvs_node.read(key="sample_0_node_1", 
                      min_version=1)
```

**Freshness Guarantee:**
- Checks: `if value.version < min_version: return None`
- Only returns: `value.version >= min_version`
- Prevents: Stale data, outdated shares

### **Key Properties**

1. **Monotonic Versions:** Versions always increase: 1 → 2 → 3 → ...
2. **Strict Comparison:** `version > current_version` (not `>=`)
3. **Quorum Enforcement:** Multiple nodes independently verify versions
4. **Freshness Guarantee:** `min_version` ensures only fresh data is read

### **Security Guarantees**

✅ **Rollback Prevention:** Older versions cannot overwrite newer versions  
✅ **Replay Prevention:** Same version cannot be written twice  
✅ **Freshness Guarantee:** Only data with `version >= min_version` is returned  
✅ **Quorum Consensus:** Multiple nodes independently verify versions  

