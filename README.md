#  TODOs
functions to port in NB

- [ ] `accuracy`, RMSE for continuous
- [x] `split_train_test`
- [ ] `variance_SSR`
- [ ] `variance_SSR_max`
- [x] `gini`
- [ ] `entropy` (not used)
- [ ] `cat_split`

# Preliminary results
In continous splitting, the naive implementation vs our implementation

500: 
— 12.80033802986145 seconds ---
--- 5.197544097900391 seconds ---

200:
--- 4.361335039138794 seconds ---
--- 1.6376280784606934 seconds ---

100:
0.8631629943847656
0.6476578712463379



The amount of time taken to construct a tree grows wrt number of trees in a forest since the implementation uses a for loop
```
(base) Violets-MacBook-Pro:math60607 vikuo$ python tree.py
--- 0.12336206436157227 seconds ---
--- 0.12372374534606934 seconds ---
--- 0.1255931854248047 seconds ---
--- 0.1471858024597168 seconds ---
--- 0.12482881546020508 seconds ---
Trees: 1
Scores: [56.09756097560976, 63.41463414634146, 60.97560975609756, 58.536585365853654, 73.17073170731707]
Mean Accuracy: 62.439%
--- 0.6273660659790039 seconds ---
--- 0.5815448760986328 seconds ---
--- 0.6488070487976074 seconds ---
--- 0.6652359962463379 seconds ---
--- 0.6279010772705078 seconds ---
Trees: 5
Scores: [70.73170731707317, 58.536585365853654, 85.36585365853658, 75.60975609756098, 63.41463414634146]
Mean Accuracy: 70.732%
--- 1.2086970806121826 seconds ---
--- 1.245736837387085 seconds ---
--- 1.275620937347412 seconds ---
--- 1.1967086791992188 seconds ---
--- 1.2417962551116943 seconds ---
Trees: 10
Scores: [75.60975609756098, 80.48780487804879, 92.6829268292683, 73.17073170731707, 70.73170731707317]
Mean Accuracy: 78.537%
```


```
#### Using threading
(base) violets-mbp:proj vikuo$ python tree.py
--- 2.0470967292785645 seconds ---
--- 2.042880058288574 seconds ---
--- 2.0631721019744873 seconds ---
--- 2.0379161834716797 seconds ---
--- 2.2384910583496094 seconds ---
Trees: 30
Scores: [80.48780487804879, 80.48780487804879, 73.17073170731707, 58.536585365853654, 85.36585365853658]
Mean Accuracy: 75.610%
--- 3.5842411518096924 seconds ---
--- 4.106899976730347 seconds ---
--- 3.554885149002075 seconds ---
--- 3.783849000930786 seconds ---
--- 3.8755860328674316 seconds ---
Trees: 50
Scores: [80.48780487804879, 87.8048780487805, 78.04878048780488, 75.60975609756098, 70.73170731707317]
Mean Accuracy: 78.537%

#### No threading
(base) violets-mbp:proj vikuo$ python tree.py
--- 3.9685792922973633 seconds ---
--- 3.8386070728302 seconds ---
--- 4.114872932434082 seconds ---
--- 4.13771390914917 seconds ---
--- 3.957604169845581 seconds ---
Trees: 30
Scores: [82.92682926829268, 75.60975609756098, 73.17073170731707, 73.17073170731707, 80.48780487804879]
Mean Accuracy: 77.073%
--- 6.425946235656738 seconds ---
--- 6.49003005027771 seconds ---
--- 6.254556179046631 seconds ---
--- 6.5482048988342285 seconds ---
--- 6.499629020690918 seconds ---
Trees: 50
Scores: [78.04878048780488, 92.6829268292683, 92.6829268292683, 73.17073170731707, 85.36585365853658]
Mean Accuracy: 84.390%
(base) violets-mbp:proj vikuo$ 
```