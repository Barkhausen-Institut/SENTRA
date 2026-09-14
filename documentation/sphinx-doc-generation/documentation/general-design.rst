General Design of Sentra
====================================

Overview
========

.. warning::

   This section documents the legacy backend-assisted deployment architecture.
   The backend implementation and web demonstrator are not part of this
   repository. Current standalone MPC training and benchmark workflows run
   without this backend; the deployment manifests require an externally
   supplied ``sentra-backend`` image.

.. plantuml::

    skinparam BackgroundColor #FFFFFF00

    participant "Sentra Backend" as SentraBackend
    participant "ACME Server (CA)" as Acme
    participant "Sentra Node 1" as SentraNode1
    participant "..." as SentraNodeX
    participant "Sentra Node n" as SentraNodeN

    group Get TLS Certificates
    SentraBackend -> Acme: Certificate Signig Request (CSR)
    SentraNode1 -> Acme: Certificate Signig Request (CSR)
    SentraNodeX -> Acme: Certificate Signig Request (CSR)
    SentraNodeN -> Acme: Certificate Signig Request (CSR)
    Acme -> SentraBackend: Certificate
    Acme -> SentraNode1: Certificate
    Acme -> SentraNodeX: Certificate
    Acme -> SentraNodeN: Certificate
    end

    group Establish GRPC Interface
    activate SentraBackend
    SentraBackend --> SentraBackend: Start GRPC endpoint using provided Certificate
    activate SentraNode1
    SentraNode1 --> SentraNode1: Start GRPC endpoint using provided Certificate
    activate SentraNode1
    SentraNodeX --> SentraNodeX: Start GRPC endpoint using provided Certificate
    activate SentraNodeN
    SentraNodeN --> SentraNodeN: Start GRPC endpoint using provided Certificate
    end

    group Node registration
    SentraNode1 -> SentraBackend: TLS Channel Establishment
    SentraNodeX -> SentraBackend: TLS Channel Establishment
    SentraNodeN -> SentraBackend: TLS Channel Establishment
    SentraBackend -> SentraNode1: Attestion request
    SentraBackend -> SentraNodeX: Attestion request
    SentraBackend -> SentraNodeN: Attestion request
    SentraNode1 -> SentraBackend: Attestion Report
    SentraNodeX -> SentraBackend: Attestion Report
    SentraNodeN -> SentraBackend: Attestion Report
    end

    group Committee selection
    SentraBackend --> SentraBackend: eligibility filtering (based on: trust score, attestation result)
    alt N_candidates < N_target
        SentraBackend -> SentraBackend: return empty candidate set
    end
    SentraBackend --> SentraBackend: committee reuse check
    alt success
        SentraBackend -> SentraBackend: return old committee
    end
    SentraBackend --> SentraBackend: greedy diversity-aware selection on sorted candidate list
    alt success
        SentraBackend -> SentraBackend: return new committee
    end
    end

    group Communicate Committee selection
    SentraBackend -> SentraNode1: Send set of Committee nodes
    SentraBackend -> SentraNodeX: Send set of Committee nodes
    SentraBackend -> SentraNodeN: Send set of Committee nodes
    end

    group Establish connections among Committee nodes
    SentraNode1 -> SentraNodeX: TLS Channel Establishment
    SentraNode1 -> SentraNodeN: TLS Channel Establishment
    SentraNodeX -> SentraNodeN: TLS Channel Establishment
    end


Architecture Overview
=====================

.. image:: figures/sentra_demonstrator.svg
   :class: with-border
   :alt: Legacy backend-assisted SENTRA architecture

Network Communication
=====================

.. image:: figures/sentra_network_comm.svg
   :class: with-border
   :alt: Network Communication in Sentra

In the legacy backend-assisted architecture, communication between the
Python ML component and the Sentra client passes through proxies in the
external backend and the Rust Sentra node. Each committee node receives a
dedicated proxy port, allocated from a configurable base port. Committee nodes
are sorted by ``node_id`` so every node derives the same port assignment.
