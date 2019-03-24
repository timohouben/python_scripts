
def identify_misfits(path, filename="results.csv"):
    """
    Identifies the model runs which have higher covariance matrix than 1 and prints input parameters.

    Parameters
    ---------_

    path : string
        Path to results
    filename : string
        "results.csv" is default
    """
    count = 0
    import pandas as pd
    df = pd.read_csv(path + "/" + filename)
    for i,item in enumerate(df["cov"][:]):
        try:
            if float(item[3:17]) > 1:
                print(item[3:17], i,df["name"][i],"S_in: ",df["S_in"][i],"T_in: ", df["T_in"][i])
                count = count + 1
        except TypeError:
            pass
        except ValueError:
            pass
    print(str(count), "misfits identified")
