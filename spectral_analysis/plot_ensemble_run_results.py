import seaborn as sns
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

data = pd.read_csv("/Users/houben/phd/results/20190513_spec_anal_hetero_ensemble_1/merge_results.csv")
# data.columns

# ax = sns.violinplot(x="obs_loc", y="T_out", hue="len_scale", data=data)
# ax = sns.violinplot(x="len_scale", y="T_out", order=[5,15,100,500], data=data)
# ax = sns.violinplot(x="obs_loc", y="T_out", hue="len_scale", order=[100,500,900,990], data=data)
#ax = sns.violinplot(x="obs_loc", y="T_out", order=[100,500,900,990], data=data, scale="width")
#ax = sns.violinplot(x="obs_loc", y="T_out", order=[100,500,900,990], data=data, scale="count")
#ax = sns.violinplot(x="obs_loc", y="T_out", order=[100,500,900,990], data=data, scale="area")
#ax = sns.violinplot(x="obs_loc", y="T_out", order=[100,500,900,990], data=data, inner="stick", scale="count")
# ax = sns.violinplot(x="obs_loc", y="T_out", order=[100,500,900,990], data=data, scale="count", scale_hue=False, bw=0.006)
#ax = sns.catplot(x="len_scale", y="T_out", col="obs_loc", order=[5,15,100, 500], data=data[data.obs_loc.isin([100, 200, 500])], kind="violin", height=3, aspect=0.7, scale="count")
#ax = sns.catplot(x="len_scale", y="T_out", col="obs_loc", order=[5,15], data=data[data.obs_loc.isin([600, 700, 800])], kind="violin", height=3, aspect=0.7, scale="count")
#ax = sns.catplot(x="obs_loc", y="T_out", order=[100,200,500,600,800], data=data[data.len_scale.isin([5,15])], kind="violin", height=5, aspect=2, scale="count", hue="len_scale", split=True)

fig = plt.figure()
sns.set_context("paper")
ax1 = sns.catplot(x="obs_loc", y="T_out",
    #order=[100,200,400,500,800],
    order=[900,920,940,990],
    data=data[data.len_scale.isin([5,15])],
    #data=data[data.len_scale.isin([100,500])],
    kind="violin",
    height=5,
    aspect=2,
    scale="count",
    hue="len_scale",
    split=True,
    legend=False)
ax2 = plt.hlines(y=data.T_in_geo.unique(),xmin=-0.5,xmax=4.5, label="geomean", color="c", alpha=0.5)
ax2 = plt.hlines(y=data.T_in_har.unique(),xmin=-0.5,xmax=4.5, label="harmean", color="r", alpha=0.005)
ax2 = plt.hlines(y=data.T_in_ari.unique(),xmin=-0.5,xmax=4.5, label="arimean", color="y", alpha=0.005)
plt.legend(loc='upper left')
ax1.set_xlabels("location of observation point [m]")
ax1.set_ylabels("$T\;[m^2/s]$ derived by fit")
plt.ylim(-0.002,0.008)
plt.savefig("/Users/houben/phd/results/20190513_spec_anal_hetero_ensemble_1/len_5_15_obs_900_920_940_990.png", dpi=300, bbox_inches="tight")




def plot_violins_for_ensemble_runs(data):
    pass
