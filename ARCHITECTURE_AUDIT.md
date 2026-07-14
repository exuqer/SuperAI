# Architecture Audit

The runtime exposes only `/api/v2`. The canonical database contains one Cloud / Space / Placement schema. Legacy V1 training, services, endpoints, migrations and compatibility synchronization were removed.
