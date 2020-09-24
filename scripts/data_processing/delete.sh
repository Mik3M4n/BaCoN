#!/bin/bash
echo "Initializing training data"
echo ""

cd "$(dirname "$0")"

cd /Users/bbose/Desktop/ML+ReACT/reactions/examples/randpk/random_pk/output


myarrayrand=(2 , 8 , 11 , 13 , 14 , 33 , 35 , 37 , 53 , 60 , 65 , 66 , 74 , 84 , 89 , 92 , 96 , 98 , 99 , 105 , 112 , 122 , 123 , 124 , 127 , 142 , 151 , 152 , 154 , 164 , 167 , 173 , 181 , 207 , 212 , 216 , 218 , 224 , 226 , 263 , 274 , 279 , 280 , 282 , 283 , 284 , 286 , 288 , 289 , 306 , 310 , 322 , 324 , 339 , 340 , 345 , 352 , 354 , 365 , 368 , 371 , 376 , 385 , 393 , 419 , 421 , 425 , 439 , 444 , 448 , 451 , 456 , 458 , 460 , 466 , 467 , 476 , 479 , 482 , 485 , 494 , 495 , 502 , 514 , 518 , 519 , 529 , 531 , 533 , 534 , 535 , 549 , 552 , 554 , 556 , 567 , 568 , 582 , 591 , 592 , 593 , 602 , 603 , 611 , 624 , 628 , 631 , 636 , 639 , 644 , 656 , 669 , 671 , 675 , 684 , 691 , 693 , 705 , 707 , 716 , 733 , 760 , 765 , 769 , 776 , 785 , 788 , 792 , 796 , 805 , 815 , 816 , 822 , 844 , 845 , 869 , 870 , 884 , 885 , 898 , 900 , 903 , 904 , 915 , 945 , 962 , 969 , 975 , 984 , 986 , 989 , 997 , 999 , 1006 , 1009 , 1011 , 1017 , 1019 , 1020 , 1031 , 1035 , 1040 , 1045 , 1050 , 1054 , 1068 , 1080 , 1089 , 1102 , 1103 , 1106 , 1120 , 1124 , 1125 , 1129 , 1140 , 1141 , 1143 , 1145 , 1154 , 1155 , 1157 , 1158 , 1163 , 1164 , 1165 , 1167 , 1170 , 1187 , 1188 , 1197 , 1199 , 1202 , 1205 , 1220 , 1228 , 1230 , 1233 , 1239 , 1244 , 1252 , 1268 , 1269 , 1275 , 1288 , 1298 , 1300 , 1302 , 1307 , 1314 , 1315 , 1318 , 1321 , 1329 , 1331 , 1336 , 1337 , 1344 , 1359 , 1367 , 1372 , 1378 , 1379 , 1382 , 1385 , 1391 , 1393 , 1394 , 1397 , 1399 , 1403 , 1414 , 1415 , 1422 , 1423 , 1438 , 1446 , 1451 , 1453 , 1455 , 1457 , 1462 , 1476 , 1477 , 1483 , 1488 , 1502 , 1503 , 1508 , 1527 , 1537 , 1541 , 1545 , 1547 , 1554 , 1577 , 1578 , 1579 , 1591 , 1594 , 1604 , 1615 , 1622 , 1625 , 1635 , 1647 , 1660 , 1662 , 1665 , 1673 , 1674 , 1675 , 1678 , 1682 , 1683 , 1685 , 1686 , 1693 , 1694 , 1699 , 1709 , 1711 , 1714 , 1724 , 1734 , 1735 , 1748 , 1749 , 1753 , 1763 , 1764 , 1775 , 1782 , 1801 , 1802 , 1808 , 1809 , 1811 , 1812 , 1821 , 1831 , 1833 , 1837 , 1842 , 1843 , 1855 , 1869 , 1876 , 1895 , 1914 , 1919 , 1921 , 1923 , 1929 , 1939 , 1964 , 1965 , 1970 , 2003 , 2005 , 2008 , 2010 , 2022 , 2025 , 2027 , 2030 , 2035 , 2050 , 2057 , 2061 , 2063 , 2066 , 2069 , 2072 , 2073 , 2086 , 2090 , 2104 , 2108 , 2109 , 2122 , 2123 , 2133 , 2136 , 2151 , 2155 , 2160 , 2162 , 2180 , 2184 , 2190 , 2191 , 2195 , 2211 , 2217 , 2223 , 2229 , 2230 , 2231 , 2242 , 2247 , 2249 , 2255 , 2265 , 2279 , 2282 , 2290 , 2299 , 2300 , 2301 , 2309 , 2316 , 2323 , 2339 , 2343 , 2355 , 2356 , 2359 , 2363 , 2364 , 2374 , 2379 , 2388 , 2401 , 2405 , 2412 , 2415 , 2419 , 2425 , 2435 , 2441 , 2445 , 2452 , 2465 , 2466 , 2473 , 2474 , 2479 , 2483 , 2488 , 2495 , 2503 , 2529 , 2530 , 2533 , 2537 , 2543 , 2549 , 2550 , 2565 , 2577 , 2586 , 2592 , 2593 , 2597 , 2614 , 2615 , 2616 , 2620 , 2627 , 2635 , 2648 , 2649 , 2694 , 2695 , 2705 , 2713 , 2719 , 2720 , 2724 , 2733 , 2740 , 2743 , 2745 , 2746 , 2750 , 2764 , 2767 , 2782 , 2789 , 2791 , 2807 , 2823 , 2827 , 2836 , 2843 , 2847 , 2866 , 2867 , 2874 , 2881 , 2891 , 2892 , 2900 , 2914 , 2923 , 2925 , 2935 , 2939 , 2958 , 2960 , 2965 , 2968 , 2974 , 2975 , 2979 , 2984 , 2986 , 2988 , 3001 , 3005 , 3014 , 3016 , 3021 , 3024 , 3026 , 3027 , 3028 , 3031 , 3058 , 3059 , 3066 , 3069 , 3073 , 3075 , 3081 , 3087 , 3089 , 3110 , 3118 , 3125 , 3127 , 3133 , 3134 , 3139 , 3152 , 3155 , 3158 , 3171 , 3173 , 3174 , 3191 , 3200 , 3201 , 3204 , 3212 , 3213 , 3216 , 3239 , 3240 , 3241 , 3242 , 3243 , 3247 , 3249 , 3252 , 3254 , 3268 , 3270 , 3271 , 3274 , 3275 , 3280 , 3285 , 3326 , 3327 , 3332 , 3342 , 3348 , 3351 , 3364 , 3378 , 3403 , 3418 , 3419 , 3428 , 3430 , 3443 , 3456 , 3457 , 3458 , 3459 , 3471 , 3478 , 3482 , 3492 , 3493 , 3509 , 3515 , 3519 , 3530 , 3531 , 3539 , 3551 , 3556 , 3557 , 3558 , 3563 , 3567 , 3579 , 3586 , 3590 , 3594 , 3597 , 3603 , 3604 , 3605 , 3621 , 3648 , 3653 , 3661 , 3662 , 3663 , 3670 , 3690 , 3702 , 3712 , 3718 , 3719 , 3721 , 3724 , 3741 , 3742 , 3753 , 3763 , 3764 , 3780 , 3791 , 3803 , 3808 , 3814 , 3824 , 3836 , 3846 , 3849 , 3861 , 3869 , 3870 , 3871 , 3873 , 3882 , 3883 , 3889 , 3896 , 3898 , 3903 , 3905 , 3906 , 3914 , 3926 , 3946 , 3962 , 3969 , 3986 , 3992 , 3998 , 4012 , 4018 , 4021 , 4029 , 4033 , 4041 , 4050 , 4052 , 4058 , 4066 , 4068 , 4074 , 4075 , 4094 , 4097 , 4103 , 4105 , 4123 , 4127 , 4130 , 4133 , 4134 , 4139 , 4140 , 4141 , 4146 , 4150 , 4162 , 4174 , 4179 , 4182 , 4189 , 4197 , 4206 , 4209 , 4212 , 4216 , 4219 , 4223 , 4224 , 4266 , 4268 , 4277 , 4279 , 4281 , 4285 , 4292 , 4295 , 4296 , 4318 , 4327 , 4331 , 4339 , 4342 , 4344 , 4355 , 4356 , 4359 , 4361 , 4366 , 4367 , 4375 , 4377 , 4389 , 4394 , 4398 , 4399 , 4403 , 4415 , 4433 , 4452 , 4453 , 4462 , 4471 , 4476 , 4483 , 4494 , 4496 , 4498 , 4506 , 4517 , 4523 , 4528 , 4535 , 4536 , 4549 , 4551 , 4561 , 4574 , 4578 , 4589 , 4592 , 4595 , 4606 , 4612 , 4630 , 4633 , 4637 , 4638 , 4639 , 4641 , 4643 , 4647 , 4654 , 4658 , 4659 , 4665 , 4677 , 4681 , 4682 , 4695 , 4699 , 4702 , 4718 , 4729 , 4732 , 4737 , 4739 , 4745 , 4748 , 4763 , 4769 , 4775 , 4792 , 4805 , 4806 , 4820 , 4828 , 4831 , 4833 , 4835 , 4840 , 4842 , 4851 , 4858 , 4867 , 4878 , 4883 , 4884 , 4896 , 4898 , 4903 , 4930 , 4935 , 4947 , 4950 , 4960 , 4961 , 4969 , 4970 , 4984 , 4993 , 4997 , 5000 , 5012 , 5028 , 5033 , 5039 , 5040 , 5042 , 5043 , 5055 , 5057 , 5059 , 5061 , 5066 , 5067 , 5072 , 5077 , 5101 , 5108 , 5110 , 5141 , 5144 , 5158 , 5166 , 5167 , 5169 , 5171 , 5173 , 5192 , 5204 , 5219 , 5221 , 5222 , 5239 , 5246 , 5248 , 5251 , 5261 , 5271 , 5280 , 5283 , 5300 , 5305 , 5306 , 5316 , 5319 , 5326 , 5332 , 5357 , 5361 , 5367 , 5371 , 5372 , 5383 , 5385 , 5396 , 5403 , 5405 , 5406 , 5408 , 5409 , 5426 , 5432 , 5446 , 5451 , 5462 , 5465 , 5473 , 5474 , 5493 , 5498 , 5507 , 5515 , 5517 , 5520 , 5537 , 5542 , 5543 , 5552 , 5554 , 5572 , 5574 , 5575 , 5581 , 5584 , 5595 , 5603 , 5607 , 5610 , 5612 , 5619 , 5620 , 5623 , 5626 , 5631 , 5640 , 5643 , 5648 , 5651 , 5667 , 5675 , 5679 , 5683 , 5687 , 5688 , 5690 , 5697 , 5706 , 5707 , 5713 , 5728 , 5738 , 5745 , 5760 , 5761 , 5763 , 5767 , 5769 , 5773 , 5776 , 5780 , 5788 , 5797 , 5798 , 5799 , 5800 , 5811 , 5821 , 5825 , 5831 , 5840 , 5848 , 5856 , 5864 , 5865 , 5883 , 5893 , 5907 , 5909 , 5919 , 5922 , 5924 , 5928 , 5936 , 5939 , 5940 , 5958 , 5961 , 5963 , 5965 , 5970 , 5973 , 5975 , 5979 , 5983 , 5991 , 5996 , 6008 , 6011 , 6014 , 6022 , 6030 , 6038 , 6039 , 6043 , 6046 , 6047 , 6070 , 6088 , 6096 , 6098 , 6102 , 6108 , 6122 , 6129 , 6131 , 6133 , 6134 , 6146 , 6151 , 6153 , 6162 , 6165 , 6171 , 6173 , 6175 , 6178 , 6186 , 6187 , 6200 , 6221 , 6266 , 6275 , 6276 , 6278 , 6279 , 6292 , 6294 , 6299 , 6300 , 6303 , 6307 , 6314 , 6324 , 6326 , 6328 , 6349 , 6352 , 6358 , 6362 , 6391 , 6393 , 6400 , 6420 , 6428 , 6431 , 6433 , 6444 , 6445 , 6448 , 6451 , 6454 , 6455 , 6458 , 6461 , 6467 , 6483 , 6492 , 6514 , 6526 , 6531 , 6575 , 6576 , 6577 , 6586 , 6588 , 6589 , 6590 , 6594 , 6601 , 6619 , 6627 , 6649 , 6653 , 6654 , 6660 , 6664 , 6671 , 6675 , 6676 , 6680 , 6682 , 6684 , 6689 , 6701 , 6702 , 6730 , 6731 , 6732 , 6746 , 6748 , 6761 , 6769 , 6773 , 6776 , 6777 , 6778 , 6783 , 6786 , 6789 , 6792 , 6811 , 6814 , 6824 , 6826 , 6833 , 6848 , 6854 , 6861 , 6868 , 6877 , 6879 , 6886 , 6892 , 6899 , 6907 , 6908 , 6913 , 6925 , 6930 , 6931 , 6939 , 6941 , 6960 , 6982 , 6983 , 6989 , 6995 , 6996 , 6999 , 7009 , 7023 , 7024 , 7028 , 7034 , 7037 , 7038 , 7047 , 7049 , 7052 , 7066 , 7070 , 7071 , 7075 , 7081 , 7082 , 7101 , 7108 , 7132 , 7135 , 7138 , 7144 , 7152 , 7154 , 7157 , 7163 , 7171 , 7172 , 7173 , 7188 , 7190 , 7204 , 7206 , 7210 , 7212 , 7216 , 7222 , 7238 , 7251 , 7253 , 7255 , 7259 , 7260 , 7263 , 7274 , 7279 , 7290 , 7298 , 7300 , 7306 , 7313 , 7332 , 7334 , 7335 , 7337 , 7342 , 7350 , 7355 , 7357 , 7358 , 7370 , 7371 , 7379 , 7396 , 7397 , 7404 , 7409 , 7412 , 7413 , 7416 , 7424 , 7430 , 7439 , 7450 , 7469 , 7484 , 7487 , 7496 , 7500 , 7505 , 7516 , 7535 , 7536 , 7551 , 7553 , 7554 , 7555 , 7556 , 7557 , 7567 , 7573 , 7577 , 7595 , 7614 , 7616 , 7617 , 7632 , 7633 , 7639 , 7640 , 7641 , 7654 , 7662 , 7666 , 7680 , 7683 , 7686 , 7689 , 7690 , 7693 , 7700 , 7705 , 7713 , 7718 , 7720 , 7730 , 7734 , 7737 , 7741 , 7742 , 7756 , 7786 , 7789 , 7793 , 7794 , 7795 , 7796 , 7798 , 7801 , 7813 , 7831 , 7837 , 7838 , 7839 , 7845 , 7865 , 7893 , 7909 , 7913 , 7914 , 7915 , 7917 , 7920 , 7921 , 7924 , 7927 , 7935 , 7950 , 7951 , 7960 , 7965 , 7981 , 7982 , 7993 , 7994 , 8000 , 8001 , 8017 , 8020 , 8032 , 8033 , 8036 , 8041 , 8074 , 8076 , 8083 , 8090 , 8092 , 8096 , 8109 , 8113 , 8121 , 8123 , 8144 , 8145 , 8172 , 8180 , 8187 , 8188 , 8189 , 8194 , 8195 , 8217 , 8233 , 8239 , 8241 , 8253 , 8254 , 8271 , 8284 , 8289 , 8292 , 8295 , 8305 , 8308 , 8316 , 8319 , 8323 , 8328 , 8329 , 8338 , 8341 , 8350 , 8352 , 8367 , 8370 , 8385 , 8391 , 8401 , 8419 , 8422 , 8424 , 8427 , 8458 , 8467 , 8470 , 8493 , 8498 , 8507 , 8508 , 8512 , 8531 , 8536 , 8539 , 8542 , 8550 , 8560 , 8574 , 8577 , 8586 , 8596 , 8598 , 8599 , 8600 , 8630 , 8631 , 8632 , 8641 , 8645 , 8648 , 8652 , 8667 , 8670 , 8674 , 8681 , 8691 , 8693 , 8702 , 8715 , 8729 , 8732 , 8742 , 8745 , 8755 , 8766 , 8768 , 8773 , 8775 , 8778 , 8787 , 8788 , 8803 , 8805 , 8806 , 8807 , 8816 , 8818 , 8823 , 8829 , 8830 , 8841 , 8846 , 8848 , 8858 , 8866 , 8869 , 8870 , 8876 , 8881 , 8905 , 8918 , 8922 , 8924 , 8941 , 8954 , 8975 , 8980 , 8983 , 8984 , 8985 , 8987 , 8992 , 8993 , 9001 , 9006 , 9011 , 9014 , 9032 , 9033 , 9034 , 9035 , 9042 , 9043 , 9046 , 9054 , 9059 , 9060 , 9061 , 9066 , 9075 , 9084 , 9085 , 9089 , 9107 , 9108 , 9117 , 9118 , 9134 , 9145 , 9146 , 9148 , 9154 , 9159 , 9172 , 9206 , 9209 , 9211 , 9236 , 9240 , 9243 , 9246 , 9253 , 9256 , 9258 , 9259 , 9263 , 9266 , 9280 , 9291 , 9311 , 9312 , 9315 , 9320 , 9321 , 9325 , 9327 , 9334 , 9335 , 9337 , 9339 , 9344 , 9355 , 9357 , 9359 , 9365 , 9367 , 9376 , 9381 , 9387 , 9392 , 9400 , 9407 , 9410 , 9416 , 9417 , 9423 , 9425 , 9426 , 9432 , 9437 , 9440 , 9448 , 9450 , 9452 , 9454 , 9456 , 9457 , 9475 , 9476 , 9493 , 9499 , 9500 , 9507 , 9511 , 9527 , 9542 , 9546 , 9553 , 9571 , 9575 , 9577 , 9586 , 9588 , 9592 , 9593 , 9614 , 9616 , 9617 , 9621 , 9626 , 9628 , 9642 , 9644 , 9653 , 9672 , 9676 , 9679 , 9681 , 9684 , 9689 , 9694 , 9695 , 9711 , 9716 , 9722 , 9724 , 9733 , 9743 , 9754 , 9755 , 9759 , 9760 , 9776 , 9792 , 9796 , 9813 , 9818 , 9822 , 9828 , 9829 , 9866 , 9870 , 9886 , 9887 , 9889 , 9893 , 9901 , 9914 , 9916 , 9917 , 9929 , 9949 , 9950 , 9951 , 9953 , 9955 , 9964 , 9967 , 9974 , 9977 , 9978 , 9986 , 9988 , 9996 , 9998 , 9999);
#printf "%s\n" "${myarray[@]}"

echo $PWD
# Start the loop over parameter values from myfile.txt
for iteration in "${myarrayrand[@]}"
do

echo "Deleting file ${iteration}"

rm  ${iteration}.txt

done

echo "DONE!!! :D "




#LC=(2 , 8 , 11 , 13 , 14 , 33 , 35 , 37 , 53 , 60 , 65 , 66 , 74 , 84 , 89 , 92 , 96 , 98 , 99 , 105 , 112 , 122 , 123 , 124 , 127 , 142 , 151 , 152 , 154 , 164 , 167 , 173 , 181 , 207 , 212 , 216 , 218 , 224 , 226 , 263 , 274 , 279 , 280 , 282 , 283 , 284 , 286 , 288 , 289 , 306 , 310 , 322 , 324 , 339 , 340 , 345 , 352 , 354 , 365 , 368 , 371 , 376 , 385 , 393 , 419 , 421 , 425 , 439 , 444 , 448 , 451 , 456 , 458 , 460 , 466 , 467 , 476 , 479 , 482 , 485 , 494 , 495 , 502 , 514 , 518 , 519 , 529 , 531 , 533 , 534 , 535 , 549 , 552 , 554 , 556 , 567 , 568 , 582 , 591 , 592 , 593 , 602 , 603 , 611 , 624 , 628 , 631 , 636 , 639 , 644 , 656 , 669 , 671 , 675 , 684 , 691 , 693 , 705 , 707 , 716 , 733 , 760 , 765 , 769 , 776 , 785 , 788 , 792 , 796 , 805 , 815 , 816 , 822 , 844 , 845 , 869 , 870 , 884 , 885 , 898 , 900 , 903 , 904 , 915 , 945 , 962 , 969 , 975 , 984 , 986 , 989 , 997 , 999 , 1006 , 1009 , 1011 , 1017 , 1019 , 1020 , 1031 , 1035 , 1040 , 1045 , 1050 , 1054 , 1068 , 1080 , 1089 , 1102 , 1103 , 1106 , 1120 , 1124 , 1125 , 1129 , 1140 , 1141 , 1143 , 1145 , 1154 , 1155 , 1157 , 1158 , 1163 , 1164 , 1165 , 1167 , 1170 , 1187 , 1188 , 1197 , 1199 , 1202 , 1205 , 1220 , 1228 , 1230 , 1233 , 1239 , 1244 , 1252 , 1268 , 1269 , 1275 , 1288 , 1298 , 1300 , 1302 , 1307 , 1314 , 1315 , 1318 , 1321 , 1329 , 1331 , 1336 , 1337 , 1344 , 1359 , 1367 , 1372 , 1378 , 1379 , 1382 , 1385 , 1391 , 1393 , 1394 , 1397 , 1399 , 1403 , 1414 , 1415 , 1422 , 1423 , 1438 , 1446 , 1451 , 1453 , 1455 , 1457 , 1462 , 1476 , 1477 , 1483 , 1488 , 1502 , 1503 , 1508 , 1527 , 1537 , 1541 , 1545 , 1547 , 1554 , 1577 , 1578 , 1579 , 1591 , 1594 , 1604 , 1615 , 1622 , 1625 , 1635 , 1647 , 1660 , 1662 , 1665 , 1673 , 1674 , 1675 , 1678 , 1682 , 1683 , 1685 , 1686 , 1693 , 1694 , 1699 , 1709 , 1711 , 1714 , 1724 , 1734 , 1735 , 1748 , 1749 , 1753 , 1763 , 1764 , 1775 , 1782 , 1801 , 1802 , 1808 , 1809 , 1811 , 1812 , 1821 , 1831 , 1833 , 1837 , 1842 , 1843 , 1855 , 1869 , 1876 , 1895 , 1914 , 1919 , 1921 , 1923 , 1929 , 1939 , 1964 , 1965 , 1970 , 2003 , 2005 , 2008 , 2010 , 2022 , 2025 , 2027 , 2030 , 2035 , 2050 , 2057 , 2061 , 2063 , 2066 , 2069 , 2072 , 2073 , 2086 , 2090 , 2104 , 2108 , 2109 , 2122 , 2123 , 2133 , 2136 , 2151 , 2155 , 2160 , 2162 , 2180 , 2184 , 2190 , 2191 , 2195 , 2211 , 2217 , 2223 , 2229 , 2230 , 2231 , 2242 , 2247 , 2249 , 2255 , 2265 , 2279 , 2282 , 2290 , 2299 , 2300 , 2301 , 2309 , 2316 , 2323 , 2339 , 2343 , 2355 , 2356 , 2359 , 2363 , 2364 , 2374 , 2379 , 2388 , 2401 , 2405 , 2412 , 2415 , 2419 , 2425 , 2435 , 2441 , 2445 , 2452 , 2465 , 2466 , 2473 , 2474 , 2479 , 2483 , 2488 , 2495 , 2503 , 2529 , 2530 , 2533 , 2537 , 2543 , 2549 , 2550 , 2565 , 2577 , 2586 , 2592 , 2593 , 2597 , 2614 , 2615 , 2616 , 2620 , 2627 , 2635 , 2648 , 2649 , 2694 , 2695 , 2705 , 2713 , 2719 , 2720 , 2724 , 2733 , 2740 , 2743 , 2745 , 2746 , 2750 , 2764 , 2767 , 2782 , 2789 , 2791 , 2807 , 2823 , 2827 , 2836 , 2843 , 2847 , 2866 , 2867 , 2874 , 2881 , 2891 , 2892 , 2900 , 2914 , 2923 , 2925 , 2935 , 2939 , 2958 , 2960 , 2965 , 2968 , 2974 , 2975 , 2979 , 2984 , 2986 , 2988 , 3001 , 3005 , 3014 , 3016 , 3021 , 3024 , 3026 , 3027 , 3028 , 3031 , 3058 , 3059 , 3066 , 3069 , 3073 , 3075 , 3081 , 3087 , 3089 , 3110 , 3118 , 3125 , 3127 , 3133 , 3134 , 3139 , 3152 , 3155 , 3158 , 3171 , 3173 , 3174 , 3191 , 3200 , 3201 , 3204 , 3212 , 3213 , 3216 , 3239 , 3240 , 3241 , 3242 , 3243 , 3247 , 3249 , 3252 , 3254 , 3268 , 3270 , 3271 , 3274 , 3275 , 3280 , 3285 , 3326 , 3327 , 3332 , 3342 , 3348 , 3351 , 3364 , 3378 , 3403 , 3418 , 3419 , 3428 , 3430 , 3443 , 3456 , 3457 , 3458 , 3459 , 3471 , 3478 , 3482 , 3492 , 3493 , 3509 , 3515 , 3519 , 3530 , 3531 , 3539 , 3551 , 3556 , 3557 , 3558 , 3563 , 3567 , 3579 , 3586 , 3590 , 3594 , 3597 , 3603 , 3604 , 3605 , 3621 , 3648 , 3653 , 3661 , 3662 , 3663 , 3670 , 3690 , 3702 , 3712 , 3718 , 3719 , 3721 , 3724 , 3741 , 3742 , 3753 , 3763 , 3764 , 3780 , 3791 , 3803 , 3808 , 3814 , 3824 , 3836 , 3846 , 3849 , 3861 , 3869 , 3870 , 3871 , 3873 , 3882 , 3883 , 3889 , 3896 , 3898 , 3903 , 3905 , 3906 , 3914 , 3926 , 3946 , 3962 , 3969 , 3986 , 3992 , 3998 , 4012 , 4018 , 4021 , 4029 , 4033 , 4041 , 4050 , 4052 , 4058 , 4066 , 4068 , 4074 , 4075 , 4094 , 4097 , 4103 , 4105 , 4123 , 4127 , 4130 , 4133 , 4134 , 4139 , 4140 , 4141 , 4146 , 4150 , 4162 , 4174 , 4179 , 4182 , 4189 , 4197 , 4206 , 4209 , 4212 , 4216 , 4219 , 4223 , 4224 , 4266 , 4268 , 4277 , 4279 , 4281 , 4285 , 4292 , 4295 , 4296 , 4318 , 4327 , 4331 , 4339 , 4342 , 4344 , 4355 , 4356 , 4359 , 4361 , 4366 , 4367 , 4375 , 4377 , 4389 , 4394 , 4398 , 4399 , 4403 , 4415 , 4433 , 4452 , 4453 , 4462 , 4471 , 4476 , 4483 , 4494 , 4496 , 4498 , 4506 , 4517 , 4523 , 4528 , 4535 , 4536 , 4549 , 4551 , 4561 , 4574 , 4578 , 4589 , 4592 , 4595 , 4606 , 4612 , 4630 , 4633 , 4637 , 4638 , 4639 , 4641 , 4643 , 4647 , 4654 , 4658 , 4659 , 4665 , 4677 , 4681 , 4682 , 4695 , 4699 , 4702 , 4718 , 4729 , 4732 , 4737 , 4739 , 4745 , 4748 , 4763 , 4769 , 4775 , 4792 , 4805 , 4806 , 4820 , 4828 , 4831 , 4833 , 4835 , 4840 , 4842 , 4851 , 4858 , 4867 , 4878 , 4883 , 4884 , 4896 , 4898 , 4903 , 4930 , 4935 , 4947 , 4950 , 4960 , 4961 , 4969 , 4970 , 4984 , 4993 , 4997 , 5000 , 5012 , 5028 , 5033 , 5039 , 5040 , 5042 , 5043 , 5055 , 5057 , 5059 , 5061 , 5066 , 5067 , 5072 , 5077 , 5101 , 5108 , 5110 , 5141 , 5144 , 5158 , 5166 , 5167 , 5169 , 5171 , 5173 , 5192 , 5204 , 5219 , 5221 , 5222 , 5239 , 5246 , 5248 , 5251 , 5261 , 5271 , 5280 , 5283 , 5300 , 5305 , 5306 , 5316 , 5319 , 5326 , 5332 , 5357 , 5361 , 5367 , 5371 , 5372 , 5383 , 5385 , 5396 , 5403 , 5405 , 5406 , 5408 , 5409 , 5426 , 5432 , 5446 , 5451 , 5462 , 5465 , 5473 , 5474 , 5493 , 5498 , 5507 , 5515 , 5517 , 5520 , 5537 , 5542 , 5543 , 5552 , 5554 , 5572 , 5574 , 5575 , 5581 , 5584 , 5595 , 5603 , 5607 , 5610 , 5612 , 5619 , 5620 , 5623 , 5626 , 5631 , 5640 , 5643 , 5648 , 5651 , 5667 , 5675 , 5679 , 5683 , 5687 , 5688 , 5690 , 5697 , 5706 , 5707 , 5713 , 5728 , 5738 , 5745 , 5760 , 5761 , 5763 , 5767 , 5769 , 5773 , 5776 , 5780 , 5788 , 5797 , 5798 , 5799 , 5800 , 5811 , 5821 , 5825 , 5831 , 5840 , 5848 , 5856 , 5864 , 5865 , 5883 , 5893 , 5907 , 5909 , 5919 , 5922 , 5924 , 5928 , 5936 , 5939 , 5940 , 5958 , 5961 , 5963 , 5965 , 5970 , 5973 , 5975 , 5979 , 5983 , 5991 , 5996 , 6008 , 6011 , 6014 , 6022 , 6030 , 6038 , 6039 , 6043 , 6046 , 6047 , 6070 , 6088 , 6096 , 6098 , 6102 , 6108 , 6122 , 6129 , 6131 , 6133 , 6134 , 6146 , 6151 , 6153 , 6162 , 6165 , 6171 , 6173 , 6175 , 6178 , 6186 , 6187 , 6200 , 6221 , 6266 , 6275 , 6276 , 6278 , 6279 , 6292 , 6294 , 6299 , 6300 , 6303 , 6307 , 6314 , 6324 , 6326 , 6328 , 6349 , 6352 , 6358 , 6362 , 6391 , 6393 , 6400 , 6420 , 6428 , 6431 , 6433 , 6444 , 6445 , 6448 , 6451 , 6454 , 6455 , 6458 , 6461 , 6467 , 6483 , 6492 , 6514 , 6526 , 6531 , 6575 , 6576 , 6577 , 6586 , 6588 , 6589 , 6590 , 6594 , 6601 , 6619 , 6627 , 6649 , 6653 , 6654 , 6660 , 6664 , 6671 , 6675 , 6676 , 6680 , 6682 , 6684 , 6689 , 6701 , 6702 , 6730 , 6731 , 6732 , 6746 , 6748 , 6761 , 6769 , 6773 , 6776 , 6777 , 6778 , 6783 , 6786 , 6789 , 6792 , 6811 , 6814 , 6824 , 6826 , 6833 , 6848 , 6854 , 6861 , 6868 , 6877 , 6879 , 6886 , 6892 , 6899 , 6907 , 6908 , 6913 , 6925 , 6930 , 6931 , 6939 , 6941 , 6960 , 6982 , 6983 , 6989 , 6995 , 6996 , 6999 , 7009 , 7023 , 7024 , 7028 , 7034 , 7037 , 7038 , 7047 , 7049 , 7052 , 7066 , 7070 , 7071 , 7075 , 7081 , 7082 , 7101 , 7108 , 7132 , 7135 , 7138 , 7144 , 7152 , 7154 , 7157 , 7163 , 7171 , 7172 , 7173 , 7188 , 7190 , 7204 , 7206 , 7210 , 7212 , 7216 , 7222 , 7238 , 7251 , 7253 , 7255 , 7259 , 7260 , 7263 , 7274 , 7279 , 7290 , 7298 , 7300 , 7306 , 7313 , 7332 , 7334 , 7335 , 7337 , 7342 , 7350 , 7355 , 7357 , 7358 , 7370 , 7371 , 7379 , 7396 , 7397 , 7404 , 7409 , 7412 , 7413 , 7416 , 7424 , 7430 , 7439 , 7450 , 7469 , 7484 , 7487 , 7496 , 7500 , 7505 , 7516 , 7535 , 7536 , 7551 , 7553 , 7554 , 7555 , 7556 , 7557 , 7567 , 7573 , 7577 , 7595 , 7614 , 7616 , 7617 , 7632 , 7633 , 7639 , 7640 , 7641 , 7654 , 7662 , 7666 , 7680 , 7683 , 7686 , 7689 , 7690 , 7693 , 7700 , 7705 , 7713 , 7718 , 7720 , 7730 , 7734 , 7737 , 7741 , 7742 , 7756 , 7786 , 7789 , 7793 , 7794 , 7795 , 7796 , 7798 , 7801 , 7813 , 7831 , 7837 , 7838 , 7839 , 7845 , 7865 , 7893 , 7909 , 7913 , 7914 , 7915 , 7917 , 7920 , 7921 , 7924 , 7927 , 7935 , 7950 , 7951 , 7960 , 7965 , 7981 , 7982 , 7993 , 7994 , 8000 , 8001 , 8017 , 8020 , 8032 , 8033 , 8036 , 8041 , 8074 , 8076 , 8083 , 8090 , 8092 , 8096 , 8109 , 8113 , 8121 , 8123 , 8144 , 8145 , 8172 , 8180 , 8187 , 8188 , 8189 , 8194 , 8195 , 8217 , 8233 , 8239 , 8241 , 8253 , 8254 , 8271 , 8284 , 8289 , 8292 , 8295 , 8305 , 8308 , 8316 , 8319 , 8323 , 8328 , 8329 , 8338 , 8341 , 8350 , 8352 , 8367 , 8370 , 8385 , 8391 , 8401 , 8419 , 8422 , 8424 , 8427 , 8458 , 8467 , 8470 , 8493 , 8498 , 8507 , 8508 , 8512 , 8531 , 8536 , 8539 , 8542 , 8550 , 8560 , 8574 , 8577 , 8586 , 8596 , 8598 , 8599 , 8600 , 8630 , 8631 , 8632 , 8641 , 8645 , 8648 , 8652 , 8667 , 8670 , 8674 , 8681 , 8691 , 8693 , 8702 , 8715 , 8729 , 8732 , 8742 , 8745 , 8755 , 8766 , 8768 , 8773 , 8775 , 8778 , 8787 , 8788 , 8803 , 8805 , 8806 , 8807 , 8816 , 8818 , 8823 , 8829 , 8830 , 8841 , 8846 , 8848 , 8858 , 8866 , 8869 , 8870 , 8876 , 8881 , 8905 , 8918 , 8922 , 8924 , 8941 , 8954 , 8975 , 8980 , 8983 , 8984 , 8985 , 8987 , 8992 , 8993 , 9001 , 9006 , 9011 , 9014 , 9032 , 9033 , 9034 , 9035 , 9042 , 9043 , 9046 , 9054 , 9059 , 9060 , 9061 , 9066 , 9075 , 9084 , 9085 , 9089 , 9107 , 9108 , 9117 , 9118 , 9134 , 9145 , 9146 , 9148 , 9154 , 9159 , 9172 , 9206 , 9209 , 9211 , 9236 , 9240 , 9243 , 9246 , 9253 , 9256 , 9258 , 9259 , 9263 , 9266 , 9280 , 9291 , 9311 , 9312 , 9315 , 9320 , 9321 , 9325 , 9327 , 9334 , 9335 , 9337 , 9339 , 9344 , 9355 , 9357 , 9359 , 9365 , 9367 , 9376 , 9381 , 9387 , 9392 , 9400 , 9407 , 9410 , 9416 , 9417 , 9423 , 9425 , 9426 , 9432 , 9437 , 9440 , 9448 , 9450 , 9452 , 9454 , 9456 , 9457 , 9475 , 9476 , 9493 , 9499 , 9500 , 9507 , 9511 , 9527 , 9542 , 9546 , 9553 , 9571 , 9575 , 9577 , 9586 , 9588 , 9592 , 9593 , 9614 , 9616 , 9617 , 9621 , 9626 , 9628 , 9642 , 9644 , 9653 , 9672 , 9676 , 9679 , 9681 , 9684 , 9689 , 9694 , 9695 , 9711 , 9716 , 9722 , 9724 , 9733 , 9743 , 9754 , 9755 , 9759 , 9760 , 9776 , 9792 , 9796 , 9813 , 9818 , 9822 , 9828 , 9829 , 9866 , 9870 , 9886 , 9887 , 9889 , 9893 , 9901 , 9914 , 9916 , 9917 , 9929 , 9949 , 9950 , 9951 , 9953 , 9955 , 9964 , 9967 , 9974 , 9977 , 9978 , 9986 , 9988 , 9996 , 9998 , 9999)


#fR=(2 , 8 , 11 , 13 , 14 , 33 , 35 , 37 , 53 , 60 , 65 , 66 , 74 , 84 , 89 , 92 , 96 , 98 , 99 , 105 , 112 , 122 , 123 , 124 , 127 , 142 , 151 , 152 , 154 , 164 , 167 , 173 , 181 , 207 , 212 , 216 , 218 , 224 , 226 , 263 , 274 , 279 , 280 , 282 , 283 , 284 , 286 , 288 , 289 , 306 , 310 , 322 , 324 , 339 , 340 , 345 , 352 , 354 , 365 , 368 , 371 , 376 , 385 , 393 , 419 , 421 , 425 , 439 , 444 , 448 , 451 , 456 , 458 , 460 , 466 , 467 , 476 , 479 , 482 , 485 , 494 , 495 , 502 , 514 , 518 , 519 , 529 , 531 , 533 , 534 , 535 , 549 , 552 , 554 , 556 , 567 , 568 , 582 , 591 , 592 , 593 , 602 , 603 , 611 , 624 , 628 , 631 , 636 , 639 , 644 , 656 , 669 , 671 , 675 , 684 , 691 , 693 , 705 , 707 , 716 , 733 , 760 , 765 , 769 , 776 , 785 , 788 , 792 , 796 , 805 , 815 , 816 , 822 , 844 , 845 , 869 , 870 , 884 , 885 , 898 , 900 , 903 , 904 , 915 , 945 , 962 , 969 , 975 , 984 , 986 , 989 , 997 , 999 , 1006 , 1009 , 1011 , 1017 , 1019 , 1020 , 1031 , 1035 , 1040 , 1045 , 1050 , 1054 , 1068 , 1080 , 1089 , 1102 , 1103 , 1106 , 1120 , 1124 , 1125 , 1129 , 1140 , 1141 , 1143 , 1145 , 1154 , 1155 , 1157 , 1158 , 1163 , 1164 , 1165 , 1167 , 1170 , 1187 , 1188 , 1197 , 1199 , 1202 , 1205 , 1220 , 1228 , 1230 , 1233 , 1239 , 1244 , 1252 , 1268 , 1269 , 1275 , 1288 , 1298 , 1300 , 1302 , 1307 , 1314 , 1315 , 1318 , 1321 , 1329 , 1331 , 1336 , 1337 , 1344 , 1359 , 1367 , 1372 , 1378 , 1379 , 1382 , 1385 , 1391 , 1393 , 1394 , 1397 , 1399 , 1403 , 1414 , 1415 , 1422 , 1423 , 1438 , 1446 , 1451 , 1453 , 1455 , 1457 , 1462 , 1476 , 1477 , 1483 , 1488 , 1502 , 1503 , 1508 , 1527 , 1537 , 1541 , 1545 , 1547 , 1554 , 1577 , 1578 , 1579 , 1591 , 1594 , 1604 , 1615 , 1622 , 1625 , 1635 , 1647 , 1660 , 1662 , 1665 , 1673 , 1674 , 1675 , 1678 , 1682 , 1683 , 1685 , 1686 , 1693 , 1694 , 1699 , 1709 , 1711 , 1714 , 1724 , 1734 , 1735 , 1748 , 1749 , 1753 , 1763 , 1764 , 1775 , 1782 , 1801 , 1802 , 1808 , 1809 , 1811 , 1812 , 1821 , 1831 , 1833 , 1837 , 1842 , 1843 , 1855 , 1869 , 1876 , 1895 , 1914 , 1919 , 1921 , 1923 , 1929 , 1939 , 1964 , 1965 , 1970 , 2003 , 2005 , 2008 , 2010 , 2022 , 2025 , 2027 , 2030 , 2035 , 2050 , 2057 , 2061 , 2063 , 2066 , 2069 , 2072 , 2073 , 2086 , 2090 , 2104 , 2108 , 2109 , 2122 , 2123 , 2133 , 2136 , 2151 , 2155 , 2160 , 2162 , 2180 , 2184 , 2190 , 2191 , 2195 , 2211 , 2217 , 2223 , 2229 , 2230 , 2231 , 2242 , 2247 , 2249 , 2255 , 2265 , 2279 , 2282 , 2290 , 2299 , 2300 , 2301 , 2309 , 2316 , 2323 , 2339 , 2343 , 2355 , 2356 , 2359 , 2363 , 2364 , 2374 , 2379 , 2388 , 2401 , 2405 , 2412 , 2415 , 2419 , 2425 , 2435 , 2441 , 2445 , 2452 , 2465 , 2466 , 2473 , 2474 , 2479 , 2483 , 2488 , 2495 , 2503 , 2529 , 2530 , 2533 , 2537 , 2543 , 2549 , 2550 , 2565 , 2577 , 2586 , 2592 , 2593 , 2597 , 2614 , 2615 , 2616 , 2620 , 2627 , 2635 , 2648 , 2649 , 2694 , 2695 , 2705 , 2713 , 2719 , 2720 , 2724 , 2733 , 2740 , 2743 , 2745 , 2746 , 2750 , 2764 , 2767 , 2782 , 2789 , 2791 , 2807 , 2823 , 2827 , 2836 , 2843 , 2847 , 2866 , 2867 , 2874 , 2881 , 2891 , 2892 , 2900 , 2914 , 2923 , 2925 , 2935 , 2939 , 2958 , 2960 , 2965 , 2968 , 2974 , 2975 , 2979 , 2984 , 2986 , 2988 , 3001 , 3005 , 3014 , 3016 , 3021 , 3024 , 3026 , 3027 , 3028 , 3031 , 3058 , 3059 , 3066 , 3069 , 3073 , 3075 , 3081 , 3087 , 3089 , 3110 , 3118 , 3125 , 3127 , 3133 , 3134 , 3139 , 3152 , 3155 , 3158 , 3171 , 3173 , 3174 , 3191 , 3200 , 3201 , 3204 , 3212 , 3213 , 3216 , 3239 , 3240 , 3241 , 3242 , 3243 , 3247 , 3249 , 3252 , 3254 , 3268 , 3270 , 3271 , 3274 , 3275 , 3280 , 3285 , 3326 , 3327 , 3332 , 3342 , 3348 , 3351 , 3364 , 3378 , 3403 , 3418 , 3419 , 3428 , 3430 , 3443 , 3456 , 3457 , 3458 , 3459 , 3471 , 3478 , 3482 , 3492 , 3493 , 3509 , 3515 , 3519 , 3530 , 3531 , 3539 , 3551 , 3556 , 3557 , 3558 , 3563 , 3567 , 3579 , 3586 , 3590 , 3594 , 3597 , 3603 , 3604 , 3605 , 3621 , 3648 , 3653 , 3661 , 3662 , 3663 , 3670 , 3690 , 3702 , 3712 , 3718 , 3719 , 3721 , 3724 , 3741 , 3742 , 3753 , 3763 , 3764 , 3780 , 3791 , 3803 , 3808 , 3814 , 3824 , 3836 , 3846 , 3849 , 3861 , 3869 , 3870 , 3871 , 3873 , 3882 , 3883 , 3889 , 3896 , 3898 , 3903 , 3905 , 3906 , 3914 , 3926 , 3946 , 3962 , 3969 , 3986 , 3992 , 3998 , 4012 , 4018 , 4021 , 4029 , 4033 , 4041 , 4050 , 4052 , 4058 , 4066 , 4068 , 4074 , 4075 , 4094 , 4097 , 4103 , 4105 , 4123 , 4127 , 4130 , 4133 , 4134 , 4139 , 4140 , 4141 , 4146 , 4150 , 4162 , 4174 , 4179 , 4182 , 4189 , 4197 , 4206 , 4209 , 4212 , 4216 , 4219 , 4223 , 4224 , 4266 , 4268 , 4277 , 4279 , 4281 , 4285 , 4292 , 4295 , 4296 , 4318 , 4327 , 4331 , 4339 , 4342 , 4344 , 4355 , 4356 , 4359 , 4361 , 4366 , 4367 , 4375 , 4377 , 4389 , 4394 , 4398 , 4399 , 4403 , 4415 , 4433 , 4452 , 4453 , 4462 , 4471 , 4476 , 4483 , 4494 , 4496 , 4498 , 4506 , 4517 , 4523 , 4528 , 4535 , 4536 , 4549 , 4551 , 4561 , 4574 , 4578 , 4589 , 4592 , 4595 , 4606 , 4612 , 4630 , 4633 , 4637 , 4638 , 4639 , 4641 , 4643 , 4647 , 4654 , 4658 , 4659 , 4665 , 4677 , 4681 , 4682 , 4695 , 4699 , 4702 , 4718 , 4729 , 4732 , 4737 , 4739 , 4745 , 4748 , 4763 , 4769 , 4775 , 4792 , 4805 , 4806 , 4820 , 4828 , 4831 , 4833 , 4835 , 4840 , 4842 , 4851 , 4858 , 4867 , 4878 , 4883 , 4884 , 4896 , 4898 , 4903 , 4930 , 4935 , 4947 , 4950 , 4960 , 4961 , 4969 , 4970 , 4984 , 4993 , 4997 , 5000 , 5012 , 5028 , 5033 , 5039 , 5040 , 5042 , 5043 , 5055 , 5057 , 5059 , 5061 , 5066 , 5067 , 5072 , 5077 , 5101 , 5108 , 5110 , 5141 , 5144 , 5158 , 5166 , 5167 , 5169 , 5171 , 5173 , 5192 , 5204 , 5219 , 5221 , 5222 , 5239 , 5246 , 5248 , 5251 , 5261 , 5271 , 5280 , 5283 , 5300 , 5305 , 5306 , 5316 , 5319 , 5326 , 5332 , 5357 , 5361 , 5367 , 5371 , 5372 , 5383 , 5385 , 5396 , 5403 , 5405 , 5406 , 5408 , 5409 , 5426 , 5432 , 5446 , 5451 , 5462 , 5465 , 5473 , 5474 , 5493 , 5498 , 5507 , 5515 , 5517 , 5520 , 5537 , 5542 , 5543 , 5552 , 5554 , 5572 , 5574 , 5575 , 5581 , 5584 , 5595 , 5603 , 5607 , 5610 , 5612 , 5619 , 5620 , 5623 , 5626 , 5631 , 5640 , 5643 , 5648 , 5651 , 5667 , 5675 , 5679 , 5683 , 5687 , 5688 , 5690 , 5697 , 5706 , 5707 , 5713 , 5728 , 5738 , 5745 , 5760 , 5761 , 5763 , 5767 , 5769 , 5773 , 5776 , 5780 , 5788 , 5797 , 5798 , 5799 , 5800 , 5811 , 5821 , 5825 , 5831 , 5840 , 5848 , 5856 , 5864 , 5865 , 5883 , 5893 , 5907 , 5909 , 5919 , 5922 , 5924 , 5928 , 5936 , 5939 , 5940 , 5958 , 5961 , 5963 , 5965 , 5970 , 5973 , 5975 , 5979 , 5983 , 5991 , 5996 , 6008 , 6011 , 6014 , 6022 , 6030 , 6038 , 6039 , 6043 , 6046 , 6047 , 6070 , 6088 , 6096 , 6098 , 6102 , 6108 , 6122 , 6129 , 6131 , 6133 , 6134 , 6146 , 6151 , 6153 , 6162 , 6165 , 6171 , 6173 , 6175 , 6178 , 6186 , 6187 , 6200 , 6221 , 6266 , 6275 , 6276 , 6278 , 6279 , 6292 , 6294 , 6299 , 6300 , 6303 , 6307 , 6314 , 6324 , 6326 , 6328 , 6349 , 6352 , 6358 , 6362 , 6391 , 6393 , 6400 , 6420 , 6428 , 6431 , 6433 , 6444 , 6445 , 6448 , 6451 , 6454 , 6455 , 6458 , 6461 , 6467 , 6483 , 6492 , 6514 , 6526 , 6531 , 6575 , 6576 , 6577 , 6586 , 6588 , 6589 , 6590 , 6594 , 6601 , 6619 , 6627 , 6649 , 6653 , 6654 , 6660 , 6664 , 6671 , 6675 , 6676 , 6680 , 6682 , 6684 , 6689 , 6701 , 6702 , 6730 , 6731 , 6732 , 6746 , 6748 , 6761 , 6769 , 6773 , 6776 , 6777 , 6778 , 6783 , 6786 , 6789 , 6792 , 6811 , 6814 , 6824 , 6826 , 6833 , 6848 , 6854 , 6861 , 6868 , 6877 , 6879 , 6886 , 6892 , 6899 , 6907 , 6908 , 6913 , 6925 , 6930 , 6931 , 6939 , 6941 , 6960 , 6982 , 6983 , 6989 , 6995 , 6996 , 6999 , 7009 , 7023 , 7024 , 7028 , 7034 , 7037 , 7038 , 7047 , 7049 , 7052 , 7066 , 7070 , 7071 , 7075 , 7081 , 7082 , 7101 , 7108 , 7132 , 7135 , 7138 , 7144 , 7152 , 7154 , 7157 , 7163 , 7171 , 7172 , 7173 , 7188 , 7190 , 7204 , 7206 , 7210 , 7212 , 7216 , 7222 , 7238 , 7251 , 7253 , 7255 , 7259 , 7260 , 7263 , 7274 , 7279 , 7290 , 7298 , 7300 , 7306 , 7313 , 7332 , 7334 , 7335 , 7337 , 7342 , 7350 , 7355 , 7357 , 7358 , 7370 , 7371 , 7379 , 7396 , 7397 , 7404 , 7409 , 7412 , 7413 , 7416 , 7424 , 7430 , 7439 , 7450 , 7469 , 7484 , 7487 , 7496 , 7500 , 7505 , 7516 , 7535 , 7536 , 7551 , 7553 , 7554 , 7555 , 7556 , 7557 , 7567 , 7573 , 7577 , 7595 , 7614 , 7616 , 7617 , 7632 , 7633 , 7639 , 7640 , 7641 , 7654 , 7662 , 7666 , 7680 , 7683 , 7686 , 7689 , 7690 , 7693 , 7700 , 7705 , 7713 , 7718 , 7720 , 7730 , 7734 , 7737 , 7741 , 7742 , 7756 , 7786 , 7789 , 7793 , 7794 , 7795 , 7796 , 7798 , 7801 , 7813 , 7831 , 7837 , 7838 , 7839 , 7845 , 7865 , 7893 , 7909 , 7913 , 7914 , 7915 , 7917 , 7920 , 7921 , 7924 , 7927 , 7935 , 7950 , 7951 , 7960 , 7965 , 7981 , 7982 , 7993 , 8000 , 8020 , 8033 , 8036 , 8076 , 8092 , 8109 , 8113 , 8123 , 8145 , 8172 , 8187 , 8194 , 8217 , 8239 , 8253 , 8254 , 8305 , 8308 , 8316 , 8319 , 8328 , 8329 , 8338 , 8341 , 8350 , 8352 , 8367 , 8370 , 8385 , 8401 , 8427 , 8467 , 8507 , 8512 , 8531 , 8536 , 8550 , 8577 , 8586 , 8596 , 8599 , 8631 , 8632 , 8641 , 8648 , 8652 , 8674 , 8681 , 8691 , 8702 , 8729 , 8732 , 8766 , 8775 , 8787 , 8805 , 8806 , 8807 , 8829 , 8848 , 8858 , 8866 , 8869 , 8870 , 8876 , 8881 , 8905 , 8918 , 8922 , 8924 , 8980 , 8984 , 8987 , 9011 , 9014 , 9032 , 9046 , 9059 , 9060 , 9061 , 9085 , 9108 , 9146 , 9148 , 9206 , 9209 , 9240 , 9243 , 9246 , 9253 , 9263 , 9280 , 9312 , 9315 , 9325 , 9327 , 9334 , 9337 , 9357 , 9359 , 9365 , 9367 , 9376 , 9381 , 9392 , 9410 , 9417 , 9425 , 9426 , 9432 , 9450 , 9452 , 9454 , 9456 , 9476 , 9493 , 9499 , 9500 , 9507 , 9527 , 9542 , 9586 , 9588 , 9614 , 9626 , 9653 , 9679 , 9681 , 9684 , 9689 , 9694 , 9711 , 9722 , 9724 , 9743 , 9796 , 9818 , 9822 , 9828 , 9829 , 9866 , 9870 , 9887 , 9916 , 9950 , 9955 , 9967 , 9988 , 9996 , 9998 , 9999)