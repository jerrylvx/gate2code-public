# Q48 minimum-weight logical basis

This is an exact quotient-space computation.  The displayed logical rows are representatives of
`rowspan(G) / rowspan(S)`; adding stabilizer rows changes representatives but not the CSS code.

- `d_X=16`, `d_Z=3`
- weight-16 logical labels: `['010', '100', '110']`
- rank of the weight-16 labels: `2`
- weight-three Z-logical columns: `[1, 17, 33]`
- its logical label: `[1, 0, 1]`
- consequence: no basis can have all three rows of weight 16; a lexicographically minimum basis has weights `[16,16,18]`.
- terminology: a projective cap is a point set with no three collinear points; in `PG(5,2)` this means no full line `{a,b,a+b}` lies inside the set.

## Coset minima

| label | min wt | # min reps | shifts |
|---|---:|---:|---|
| `001` | 18 | 8 | `010000`, `010101`, `011011`, `011101`, `110010`, `110110`, `111000`, `111111` |
| `010` | 16 | 2 | `111110`, `111111` |
| `011` | 18 | 16 | `000001`, `000111`, `001000`, `001101`, `010011`, `010100`, `011001`, `011101`, `100011`, `100100`, `101011`, `101111`, `110010`, `110100`, `111001`, `111100` |
| `100` | 16 | 6 | `000100`, `000111`, `001001`, `001011`, `101000`, `101001` |
| `101` | 18 | 8 | `000011`, `000101`, `001010`, `001111`, `100011`, `100100`, `101011`, `101111` |
| `110` | 16 | 2 | `010110`, `010111` |
| `111` | 18 | 16 | `000011`, `000110`, `001010`, `001100`, `010000`, `010100`, `011010`, `011101`, `100000`, `100100`, `101000`, `101111`, `110000`, `110101`, `111011`, `111101` |

## Minimum quotient-basis rows

```
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 1 0 0 1 0 1 1 0 0 0 1 1 1 0 0 1 1 1 0 0 1 1 0 0 1 0 0 1
0 1 0 1 1 0 1 0 1 0 1 0 0 1 0 1 1 1 1 1 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
1 0 0 0 1 0 0 0 1 1 0 1 0 0 1 0 0 1 1 1 0 1 0 0 0 1 0 0 1 0 0 0 0 1 1 0 1 0 1 0 0 0 0 0 1 1 0 0
```

## Comparison with cap normal form

| representative | K row weights | K-fiber sizes | cap fibers | projective lines inside fibers | CH | single-qubit transversal |
|---|---|---|---:|---:|---|---|
| original K | [24, 20, 30] | `5,11,3,5,5,7,5,7` | 6 | 3 | True | True |
| cap-normal K | [24, 24, 26] | `5,7,7,5,5,7,5,7` | 8 | 0 | True | True |
| minimum-weight quotient basis | [16, 16, 18] | `11,9,7,5,9,3,3,1` | 7 | 4 | True | True |

The cap-normal representative has no projective line in any logical-label fiber. A minimum-weight logical basis is a separate choice.
