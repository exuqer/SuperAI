# Snapshots

Every run stores an immutable initial snapshot and one post-settle snapshot per completed step. Each snapshot has a canonical state hash, state delta, events and clusters. Restore is an explicit API operation; exporting or comparing snapshots is read-only.

