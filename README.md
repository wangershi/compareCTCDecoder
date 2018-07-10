# compareCTCDecoder
Most codes are from [CTC.ipynb](https://github.com/DingKe/ml-tutorial/blob/master/ctc/CTC.ipynb) which is written in Chinese.

Compare three CTC decoder, that is greedy decoder, beam decoder and prefix beam decoder.

I give a example of the network outputs, which is processed by SoftMax, '0','1','2' represents the label and '0' represents blank.
| name  | t=1   | t=2   | t=3 |
|:--------:|:------|--------:|
| 0     | 0.25  | 0.4   | 0.1 |
| 1     | 0.4   | 0.35  | 0.5 |
| 2     | 0.35  | 0.25  | 0.4 |

# raw decode
Beacause there are 3 labels and T=3, so there will be 3^3=27 paths, you can use beamDecode() and set the beamsize be 27 to get all paths, that is:
| path      | score |
|:--------:|--------:|
| (2, 1)    | 0.2185
| (1, 2)    | 0.2050
| (1,)      | 0.2025
| (2,)      | 0.1290
| (1, 1)    | 0.0800
| (2, 2)    | 0.0560
| (1, 2, 1) | 0.0500
| (2, 1, 2) | 0.0490
| ()        | 0.0100

As we can see, top-path is (2, 1), it's score is 0.2185.

# greedy decode
Path is (1, 0, 1), after many-to-one map, the path is (1, 1) which is different from top-path in raw decode, and the score is 0.08 which is lower than score in raw decode.

# beam decode
Top-path is (1, 0, 1), after many-to-one map, the path is (1, 1) which is different from top-path in raw decode, and the score is 0.08 which is lower than score in raw decode.

# prefix beam decode
Top-path is (1, 2), after many-to-one map, the path is (1, 2) which is same from top-path in raw decode, and the score is 0.12 which is lower than score in raw decode.
So, obviously, prefix beam decode is better than greedy decode and beam decode.
And the reason why score is lower than score in raw decode is that I set the beamSize be 2, if beamSize=3, the score will 0.2178, which is same with the score in raw decode.


