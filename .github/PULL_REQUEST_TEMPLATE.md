## Summary

- what changed:
- why it changed:

## Verification

- [ ] `cargo fmt`
- [ ] `cargo test`
- [ ] `cargo test --features "candle burn"`
- [ ] `cargo clippy --all-targets --features "candle burn" -- -D warnings`
- [ ] `python3 -m py_compile scripts/true_pre_post_common.py scripts/true_pre_post_pytorch.py scripts/true_pre_post_mlx.py`

## Benchmark impact

- [ ] no benchmark contract change
- [ ] benchmark contract changed and documented

## Notes

Add any caveats, follow-up work, or release notes here.
