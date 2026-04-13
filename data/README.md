# Benchmark data

Each problem is four `.dat` files: `Z` (CSR matrix as COO triplets), `rhs`, `Dv`, `Dl` (known solution, `Dl` is negated when concatenated — see `utils.h::readKnownSolution`).

## Layout

1. `ancf/refine<L>/<S>/` — single-rig ANCF, tracked. `<L>` = refinement level (1 or 2); `<S>` = spoke count (16 or 80). Filename prefix `2002`. Currently present: `refine1/{16,80}` and `refine2/80`.
2. `ancf/multi_rig/<N>_rigs/` — multi-rig ANCF, **not tracked** (gitignored). `<N>` = rig count ∈ {1, 2, 4, 8, 10, 25, 50}. Filename prefix `201`.
3. `mbs/` — multibody dynamics, tracked. Filename prefix `26`.

## `ancf/multi_rig/` (local only)

| N    | matrix size `n` | nnz          |
| ---- | --------------- | ------------ |
|  1   |      24,687     |    2,095,539 |
|  2   |      49,374     |    4,191,078 |
|  4   |      98,748     |    8,382,156 |
|  8   |     197,496     |   16,764,310 |
| 10   |     246,870     |   20,955,390 |
| 25   |     617,175     |   52,388,465 |
| 50   |   1,234,350     |  104,776,935 |

### Download

Archive (tar.gz) containing the four canonical files per rig count:

- **Box:** `<https://uwmadison.box.com/s/85wzzsl38joxmj2jgc5oaliqh7k8ah4n>`

Extract at the repo root