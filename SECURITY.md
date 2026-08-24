# Vaddro Security Policy

[中文版](SECURITY_CN.md)

## Security model

Vaddro separates Seed custody from GPU computation:

- The offline machine keeps the Seed private scalar `k_seed`.
- A GPU worker receives only the corresponding public point, search rule, and shard configuration.
- On a hit, the worker returns a Search Share, Variant, address, and public verification metadata.
- The final private key is derived and verified only on the offline machine.

~~~text
Offline trusted machine                         Online GPU workers
-----------------------                         ------------------
seed.py
  -> Seed private scalar k_seed (never leaves)
  -> public point P_seed  ---------------------> P_seed + search rule + shard
                                                  CUDA/PTX search
verify_result.py  <----------------------------- Share + Variant + public metadata
combine.py
  -> k_seed + verified Share
  -> final private key and independently verified EVM address
~~~

Under the secp256k1 Elliptic Curve Discrete Logarithm assumption, the worker cannot recover the Seed private scalar from its public point. Split-Key does not protect a compromised offline machine, a leaked Seed or wallet, a malicious build, or an implementation flaw.

CUDA startup self-tests, runtime canaries, host-side hit reconstruction, and `verify_result.py` detect computational errors. They do not replace release verification or independent offline validation.

### Sharding security boundary

Deterministic sharding divides the 48-bit high-limb namespace of the random base Search Share into contiguous ranges:

~~~text
width       = 2^48 / SHARD_COUNT
shard_begin = width × SHARD_INDEX
shard_end   = shard_begin + width
~~~

- Workers on one task use the same power-of-two `SHARD_COUNT` from 1 through 65536 and a unique zero-based `SHARD_INDEX`.
- Unique active indices prevent overlap of the base Search Share ranges. Vaddro does not centrally assign or coordinate those indices.
- Shard metadata is recorded in state, JSONL events, and result files. Public verification rejects a tagged Share outside its declared range.
- A restart chooses a new random starting point inside the assigned shard. Sharding coordinates concurrent workers but is not persistent checkpointing or a record of historical coverage.
- GLV/negation variants are derived after base-space partitioning. Equivalent effective candidates across shards remain theoretically possible with cryptographically negligible probability.

## Required practices

- Run `seed.py` and `combine.py` only on an isolated, trusted machine.
- Never copy a Seed, final private key, or wallet file to a GPU worker, repository, shared filesystem, log, or support report.
- Verify the release checksum and build from the intended tagged source.
- Run `verify_result.py` before moving a result into the offline environment.
- Independently confirm the final address and complete a small-value send/return test before production use.
- Protect offline backups and private files; use restrictive permissions such as `0600` where supported.
- Publish official addresses through an independently controlled channel. A vanity pattern alone does not prove ownership.

## Supported versions

Only the latest tagged release receives security fixes. Development snapshots and untagged binaries are not supported.

## Reporting a vulnerability

Report security or cryptographic vulnerabilities privately:

- [GitHub Private Security Advisory](https://github.com/manifoldor/vaddro/security/advisories/new)
- Email: yishan ([linyishan@gmail.com](mailto:linyishan@gmail.com))

Include a clear description, reproduction steps, and a proposed remediation if available. Use synthetic test material and never include a production Seed, private key, wallet, or unredacted secret.

Do not open a public issue for a critical vulnerability before a fix and disclosure have been coordinated.
