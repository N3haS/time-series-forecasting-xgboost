# time-series-forecasting-xgboost
Time series forecasting for networking datasets using the binary model XGBoost. In this, we first make a main model for the dataset. After this, we conduct beta testing on the model by the means of a concept called Chaos Engineering. This basically states that, to understand how the model would work incase of tough situations, we inject chaos intentionallly and test the models' ability to bounce back and work normally after the outage. This is completely coded in Python. The results for the same are shown below respectively. 

1) Main Model 
The model's outputs are as follows:

MAE: 0.8940828452220706
RMSE: 7.68891718826303
R² Accuracy: 0.9980525713595094
      Epoch       MAE      RMSE  Training Loss (RMSE)
0         1  0.894083  7.688917            122.443963
1         2  0.894083  7.688917             86.205293
2         3  0.894083  7.688917             60.856524
3         4  0.894083  7.688917             43.212456
4         5  0.894083  7.688917             30.962986
...     ...       ...       ...                   ...
1995   1996  0.894083  7.688917              7.688917
1996   1997  0.894083  7.688917              7.688917
1997   1998  0.894083  7.688917              7.688917
1998   1999  0.894083  7.688917              7.688917
1999   2000  0.894083  7.688917              7.688917

[2000 rows x 4 columns]
                             ds         yhat
0 2019-04-08 07:07:33.402182144  1448.213989
1 2019-04-08 07:08:33.402182144  1448.213989
2 2019-04-08 07:09:33.402182144  1448.213989
3 2019-04-08 07:10:33.402182144  1448.213989
4 2019-04-08 07:11:33.402182144  1448.213989

The R² accuracy was of 99.8% which is great!

2) Chaos Engineering - Traffic Volume
Here there are two parts to it, 
  i) Traffic Drop
    The drop was by half, that is, 50%. This yielded the following results:
    MAE: 0.4470428574253136
    RMSE: 3.844463345163432
    Baseline MAE: 31.20467892098353
    Baseline RMSE: 92.5373227368704  
          Epoch        MAE       RMSE  Training Loss (RMSE)
    0         1  20.113734  61.221982             61.221982
    1         2  14.139664  43.102647             43.102647
    2         3   9.959057  30.428262             30.428262
    3         4   7.039292  21.606228             21.606228
    4         5   5.003573  15.481493             15.481493
    ...     ...        ...        ...                   ...
    1995   1996   0.447043   3.844463              3.844463
    1996   1997   0.447043   3.844463              3.844463
    1997   1998   0.447043   3.844463              3.844463
    1998   1999   0.447043   3.844463              3.844463
    1999   2000   0.447043   3.844463              3.844463
    [2000 rows x 4 columns]
    
    Average Percentage Error in MAE: 0.38%
    Mean Percentage Change in MAE (MPCM-MAE): 0.17%
    Chaos MAE: 20.113734345334787
    Change in MAE: 30.757636063558216
    Baseline MAE : 31.20467892098353
    Time to Return to Baseline MAE: 1900 epochs

    ![image](https://github.com/user-attachments/assets/da8ffe6d-50b0-4fe2-b468-b807f19011a0)

                               time  forecasted_altered_traffic_volume
    0 1970-01-01 00:00:01.554707253                         723.999939
    1 1970-01-01 00:01:01.554707253                         723.999939
    2 1970-01-01 00:02:01.554707253                         723.999939
    3 1970-01-01 00:03:01.554707253                         723.999939
    4 1970-01-01 00:04:01.554707253                         723.999939
  ii) Traffic Surge
    The surge was by a total of 150%. This yielded the below results:
    MAE: 1.3411905219031848
    RMSE: 11.533376842754388
    Baseline MAE: 93.61403676295059
    Baseline RMSE: 277.6119682106112
          Epoch        MAE        RMSE  Training Loss (RMSE)
    0         1  60.341335  183.665954            183.665954
    1         2  42.419028  129.307935            129.307935
    2         3  29.877205   91.284767             91.284767
    3         4  21.117896   64.818687             64.818687
    4         5  15.010744   46.444496             46.444496
    ...     ...        ...         ...                   ...
    1995   1996   1.341191   11.533377             11.533377
    1996   1997   1.341191   11.533377             11.533377
    1997   1998   1.341191   11.533377             11.533377
    1998   1999   1.341191   11.533377             11.533377
    1999   2000   1.341191   11.533377             11.533377
    [2000 rows x 4 columns]
    
    Average Percentage Error in MAE: 0.38%
    Mean Percentage Change in MAE (MPCM-MAE): 0.17%
    Chaos MAE: 60.34133478539404
    Change in MAE: 33.27270197755655
    Baseline MAE : 93.61403676295059
    Time to Return to Baseline MAE: 1906 epochs

    ![image](https://github.com/user-attachments/assets/f25960de-c5a0-42a0-978d-a78f47ccfbd8)
                              time  forecasted_altered_data_len(yhat)
    0 1970-01-01 00:00:01.554707253                        2171.999756
    1 1970-01-01 00:01:01.554707253                        2171.999756
    2 1970-01-01 00:02:01.554707253                        2171.999756
    3 1970-01-01 00:03:01.554707253                        2171.999756
    4 1970-01-01 00:04:01.554707253                        2171.999756
    
3) Chaos Engineering - Packet Size Distribution
The way I dealt with this is by increasing the proportion of the smaller packets, and this resulted in the results below:

MAE: 0.7152668919566719
RMSE: 6.151143066150975
Baseline MAE: 49.927486273573734
Baseline RMSE: 148.05971637899268
      Epoch        MAE       RMSE  Training Loss (RMSE)
0         1  32.181968  97.955170             97.955170
1         2  22.623500  68.964238             68.964238
2         3  15.934538  48.685227             48.685227
3         4  11.262914  34.569974             34.569974
4         5   8.005740  24.770392             24.770392
...     ...        ...        ...                   ...
1995   1996   0.715305   6.151143              6.151143
1996   1997   0.715305   6.151143              6.151143
1997   1998   0.715305   6.151143              6.151143
1998   1999   0.715305   6.151143              6.151143
1999   2000   0.715305   6.151143              6.151143

[2000 rows x 4 columns]
Average Percentage Error in MAE: 0.38%
Mean Percentage Change in MAE (MPCM-MAE): 0.17%
Chaos MAE: 32.18196754555738
Change in MAE: 17.745518728016357
Baseline MAE : 49.927486273573734
Time to Return to Baseline MAE: 1941 epochs

![image](https://github.com/user-attachments/assets/3bb04c9b-7c42-43ca-8240-ac44ed58d4ee)

                           time  forecasted_altered_data_len(yhat)
0 1970-01-01 00:00:01.554707253                        1158.399902
1 1970-01-01 00:01:01.554707253                        1158.399902
2 1970-01-01 00:02:01.554707253                        1158.399902
3 1970-01-01 00:03:01.554707253                        1158.399902
4 1970-01-01 00:04:01.554707253                        1158.399902


5) Chaos Engineering - Flow Rate
Adding jitters which is random small variations in the network caused a lot of changes to the results, which can be seen below:

MAE: 173.34753446163498
RMSE: 190.49404523837077
Baseline MAE: 2055724.8973953375
Baseline RMSE: 2594818.538324874
    Epoch           MAE          RMSE  Training Loss (RMSE)
0       1  1.071169e+06  1.108376e+06          1.108376e+06
1       2  7.498408e+05  7.758905e+05          7.758905e+05
2       3  5.249365e+05  5.431752e+05          5.431752e+05
3       4  3.674525e+05  3.802127e+05          3.802127e+05
4       5  2.572054e+05  2.661455e+05          2.661455e+05
..    ...           ...           ...                   ...
92     93  1.672342e+02  1.889129e+02          1.889129e+02
93     94  1.672342e+02  1.889129e+02          1.889129e+02
94     95  1.672342e+02  1.889129e+02          1.889129e+02
95     96  1.672342e+02  1.889129e+02          1.889129e+02
96     97  1.672342e+02  1.889129e+02          1.889129e+02

[97 rows x 4 columns]
Average Percentage Error in MAE: 0.00001116%
Mean Percentage Change in MAE (MPCM-MAE): 7.69%
Chaos MAE: 1071169.280305562
Change in MAE: 984555.6170897754
Baseline MAE : 2055724.8973953375
Time to Return to Baseline MAE: 51 epochs

![image](https://github.com/user-attachments/assets/ed35d77f-e0e0-4cfb-866c-b2870f5ba51b)

Forecasted Time: [1.5547069e+09 1.5547069e+09 1.5547069e+09 ... 1.5514342e+09 1.5547069e+09
 1.5547069e+09]
 OR
                          time  forecasted_time_data(yhat)
0 1970-01-01 00:00:01.554707253                1.554707e+09
1 1970-01-01 00:01:01.554707253                1.554707e+09
2 1970-01-01 00:02:01.554707253                1.554707e+09
3 1970-01-01 00:03:01.554707253                1.554707e+09
4 1970-01-01 00:04:01.554707253                1.554707e+09

7) Chaos Engineering - Port Usage
The chaos was injected by redistributing traffic for targetted ports. Results are seen as follows:
MAE: 1739.3733527355873
RMSE: 3751.4266937776806
Baseline MAE: 1553485834.903213
Baseline RMSE: 1553486641.850082
    Epoch          MAE         RMSE  Training Loss (RMSE)
0       1  2238.855509  4159.419317           4159.419317
1       2  2083.410532  3970.538021           3970.538021
2       3  1975.091895  3869.480826           3869.480826
3       4  1902.018718  3817.072251           3817.072251
4       5  1850.987591  3789.596029           3789.596029
..    ...          ...          ...                   ...
79     80  1739.373357  3751.426692           3751.426692
80     81  1739.373357  3751.426692           3751.426692
81     82  1739.373357  3751.426692           3751.426692
82     83  1739.373357  3751.426692           3751.426692
83     84  1739.373357  3751.426692           3751.426692

[84 rows x 4 columns]
(104722, 7)
           time  proto  data_len  ip_src  ip_dst  src_port  dst_port
0  1.551431e+09      6        63       0       0       443     59158
1  1.551431e+09      6        63       1       0       443     44104
2  1.551431e+09      6       136       2       1     53954       443
3  1.551431e+09      6        46       2       1     53954       443
4  1.551431e+09      6       137       3       0       443     53954
Average Percentage Error in MAE: 0.00001114%
Mean Percentage Change in MAE (MPCM-MAE): 0.30%
Chaos MAE: 2238.855508847577
Change in MAE: 1553483596.0477042
Baseline MAE : 1553485834.903213
Time to Return to Baseline MAE: 62 epochs
Minimum MAE: 1739.3339947257698
Maximum MAE: 2238.855508847577

![image](https://github.com/user-attachments/assets/90bac62a-9e97-4c70-9207-fb1613e7b9ee)
                           time   forecast
0 2019-04-08 07:08:33.401565184  23848.625
1 2019-04-08 07:09:33.401565184  23848.625
2 2019-04-08 07:10:33.401565184  23848.625
3 2019-04-08 07:11:33.401565184  23848.625
4 2019-04-08 07:12:33.401565184  23848.625

  
