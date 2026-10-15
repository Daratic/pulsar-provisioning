# pulsar-provisioning

Install scripts and agent code for Pulsar's rented GPU instances.

Vast.ai's base image fetches `<runner-set>.sh` from a pinned commit of this
repository through its `PROVISIONING_SCRIPT` variable, and the script fetches
`agent/` from the same commit. Nothing here is secret, and nothing here names
a private network address. A push to this repository changes nothing until
Pulsar's own configuration moves its pin to the new commit.

Every file here is generated. Do not edit it here: the source is Pulsar's
`app/model_adapters/runner_catalog.py` and `agent/`, exported by Pulsar's
`scripts/export_provisioning_repo.py`.
