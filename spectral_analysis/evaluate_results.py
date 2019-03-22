
def identify_missfits(path):
    """
    Identifies the model runs which have higher covariance matrix than 1 and prints input parameters.

    Parameters
    ---------_

    path : string
        Path to results
    """
    import pandas as pd
    df = pd.read_csv(path + "/" + "results.csv")
    for i,item in enumerate(df["cov"][:]):
        try:
        	if float(item[3:17]) > 1:
    	    print(item[3:17], i,df["name"][i],"S_in: ",df["S_in"][i],"T_in: ", df["T_in"][i])
        except TypeError:
            pass
        except ValueError:
            pass
