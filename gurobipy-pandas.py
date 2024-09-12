# Databricks notebook source
# MAGIC %md
# MAGIC https://support.gurobi.com/hc/en-us/articles/20660169766545-Databricks-Installation

# COMMAND ----------

import os

path = os.getenv("GRB_LICENSE_FILE")
if path is None:
    print("Gurobi license path not set")
else:
    print(f"Expected license in {path}")
    print("Found" if os.path.isfile(path) else "Not found")

# COMMAND ----------

sdf_asset_counts = spark.table("dlr_change.asset_count_eligibility")
sdf_cons_store = spark.table("dlr_change.consultant_store_eligibility")
sdf_cons_string = spark.table("dlr_change.consultant_string_eligibility")
sdf_store_dim = spark.table("dlr_change.base_store_current_dim")
sdf_string_raw = spark.table("dlr_change.strings")

# COMMAND ----------

import os

import gurobipy as gp
import gurobipy_pandas as gppd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from gurobipy import GRB


# set options to display all data

pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)


if __name__ == "__main__":
    # 1. store-dim data
    df_store_dim = sdf_store_dim.toPandas().query(
        '(banner_cd == "CTR" or banner_cd == "PTY") and store_status_cd == "A"'
    )

    # 2. string raw
    df_string_raw = sdf_string_raw.toPandas().dropna(subset=["sqft", "date_in_store"])

    # 3.consultant-string-eligibility
    df_consultant_string_raw = sdf_cons_string.toPandas()
    before_filter = set(df_consultant_string_raw["string_id"])

    print(
        f"Number of Strings in Consultant-String Table: {df_consultant_string_raw['string_id'].nunique()}",
    )

    print(
        f"Number of Resources in Consultant-String Table:{df_consultant_string_raw['resource_id'].nunique()}",
    )

    # 4.consultant-store-eligibility
    df_consultant_store_raw = sdf_cons_store.toPandas()

    before_filter_co = set(df_consultant_store_raw["store_num"])

    print(
        f"Number of Strings in Consultant-Store Table: {df_consultant_store_raw['string_id'].nunique()}",
    )
    print(
        f"Number of Stores in Consultant-Store Table:{df_consultant_store_raw['store_num'].nunique()}",
    )
    print(
        f"Number of Resources in Consultant-Store Table:{df_consultant_store_raw['resource_id'].nunique()}",
    )

    # 5.asset-count-eligibility
    df_asset_counts_raw = sdf_asset_counts.toPandas()

    before_filter_ac = set(df_asset_counts_raw["store_num"])

    print(
        f"Number of Strings in Asset Counts Table:{df_asset_counts_raw['string_id'].nunique()}",
    )
    print(
        f"Number of Stores in Asset Counts Table:{df_asset_counts_raw['store_num'].nunique()}",
    )
    print(
        f"Number of Resources in Asset Counts Table:{df_asset_counts_raw['resource_id'].nunique()}",
    )

# COMMAND ----------

model = gp.Model()

# Handy trick for live coding, not for production
gppd.set_interactive()

# Quite please
gp.setParam("OutputFlag", 0)

print("-" * 50)
# *****************************************************************************************************************
#                                       1. Decision Variables
# *****************************************************************************************************************
# Define Decision Variables for Yis - Equals 1 if consultant i is assigned to string s, 0 otherwise
df_consultant_string = (
    df_consultant_string_raw.reset_index(
        drop=True,
    )
    .drop_duplicates(subset=["resource_id", "string_id"])
    .set_index(["resource_id", "string_id"])
)

y = gppd.add_vars(
    model,
    df_consultant_string,
    vtype=GRB.BINARY,
    name="y",
)

after_filter = set(y.index.get_level_values("string_id"))
print(f"Number of Unique Strings in Yis Variable: {len(after_filter)}")
print(
    f"No Consultant for String IDs: {before_filter.difference(after_filter)}",
)

# Define Decision Variables for Xik - Equals 1 if consultant i is assigned to store k, 0 otherwise
df_consultant_store_X_ik = (
    df_consultant_store_raw.query(
        "resource_is_eligible_for_store_changeover == 1",
    )
    .reset_index(
        drop=True,
    )
    .drop_duplicates(subset=["resource_id", "store_num"])
    .set_index(["resource_id", "store_num"])
    .gppd.add_vars(
        model,
        vtype=GRB.BINARY,
        name="x",
    )
)
after_filter_co = set(
    df_consultant_store_X_ik.index.get_level_values("store_num"),
)
print("\n")
print(f"Number of Unique Stores in Xik Variable: {len(after_filter_co)}")
print(
    "Number of Unique Strings in Xik Variable",
    df_consultant_store_X_ik["string_id"].nunique(),
)
print(
    f"No Consultant for Store Nums: {before_filter_co.difference(after_filter_co)}",
)
print("\n")

# Define Decision Variables for Uikn - Equals 1 if consultant or analyst i is assigned to store k on asset count week n, 0 otherwise
df_asset_counts_U_ikn = (
    df_asset_counts_raw.query(
        "resource_is_eligible_for_asset_counts == 1 and is_asset_counting_week == 1 and at_least_one_consultant_available == 1",
    )
    .reset_index(
        drop=True,
    )
    .drop_duplicates(subset=["resource_id", "store_num", "week_id"])
    .set_index(["resource_id", "store_num", "week_id"])
    .gppd.add_vars(
        model,
        vtype=GRB.BINARY,
        name="u",
    )
)
after_filter_ac = set(
    df_asset_counts_U_ikn.index.get_level_values("store_num"),
)
print(f"Number of Unique Stores in Uikn: {len(after_filter_ac)}")
print(
    "Number of Unique Strings in Uikn: ",
    df_asset_counts_U_ikn["string_id"].nunique(),
)
print(
    f"No Resource is Available for Store Asset Counts: {before_filter_ac.difference(after_filter_ac)}",
)
print("-" * 50)

# Define Decision Variables for zkn - Number of extra resources required to satisfy the requirements of store k asset listing on week n
df_asset_counts_Z_kn = (
    df_asset_counts_raw.query(
        "resource_is_eligible_for_asset_counts == 1 and is_asset_counting_week == 1 and at_least_one_consultant_available == 1",
    )
    .reset_index(
        drop=True,
    )
    .drop_duplicates(subset=["store_num", "week_id"])
    .set_index(["store_num", "week_id"])
)
z = gppd.add_vars(
    model,
    df_asset_counts_Z_kn,
    vtype=GRB.INTEGER,
    name="z",
)

# *****************************************************************************************************************
#                                     2. Parameters
# *****************************************************************************************************************

# Parameter D_k to show the staffing required at each store k
df_resource_weeks_D_k = (
    df_asset_counts_raw.query(
        "resource_is_eligible_for_asset_counts == 1 and is_asset_counting_week == 1 and at_least_one_consultant_available == 1",
    )
    .reset_index(
        drop=True,
    )
    .drop_duplicates(subset=["store_num"])
    .set_index(["store_num"])
)

# Parameter r_ik to show travel distance between resource i and store k
df_travel_distance_r_ik = (
    df_asset_counts_raw.query(
        "resource_is_eligible_for_asset_counts == 1 and is_asset_counting_week == 1 and at_least_one_consultant_available == 1",
    )
    .reset_index(
        drop=True,
    )
    .drop_duplicates(subset=["resource_id", "store_num"])
    .set_index(["resource_id", "store_num"])
)

# *****************************************************************************************************************
#                                       3. Auxilary Decision Variables
# *****************************************************************************************************************

# Define Aux Decision Variable for tik - equals to 1 if resource_id i is assigned to asset counts/co for store k
t = gppd.add_vars(
    model,
    df_travel_distance_r_ik,
    vtype=GRB.BINARY,
    name="t",
)

# Define Auxilary Decision Variable - w total extra resources
w = model.addVar(vtype=GRB.INTEGER, name="w")

# Define auxiliary variable called total_assignment to count the workload for each resource i
df_asset_counts_T_i = (
    df_asset_counts_raw.query(
        "resource_is_eligible_for_asset_counts == 1 and is_asset_counting_week == 1 and at_least_one_consultant_available == 1",
    )
    .reset_index(
        drop=True,
    )
    .drop_duplicates(subset=["resource_id"])
    .set_index(["resource_id"])
)
total_assignments = gppd.add_vars(
    model,
    df_asset_counts_T_i,
    vtype=GRB.INTEGER,
    name="total_assignments",
)

# Min and Max Number of Assignments
minAssignment = model.addVar(name="minAssign")
maxAssignment = model.addVar(name="maxAssign")

# ************************************************************************************************************************
#                                                4. Constraints - Business Operations
# ************************************************************************************************************************

# Constraint 1:  For each store asset count staffing requirements must be met or else third party staff is hired
# sum_n(z_kn + sum_i(u_ikn)) = D_k for all k
staffing_requirements = gppd.add_constrs(
    model,
    z.groupby("store_num").sum()
    + df_asset_counts_U_ikn["u"].groupby("store_num").sum(),
    GRB.EQUAL,
    df_resource_weeks_D_k["num_resource_weeks_needed"],
    name="staffing_requirements",
)

# print (staffing_requirements)
# print (staffing_requirements.apply(model.getRow))
# print (staffing_requirements.gppd.RHS)

# Constraint 1b:  Ensures that analyst is not there alone on a given asset count week and store. There is at least one consultant
M = 1000
LHS, RHS = (
    df_asset_counts_U_ikn["u"]
    .groupby(["store_num", "week_id"])
    .sum()
    .align(
        df_asset_counts_U_ikn.query("resource_is_consultant == 1")["u"]
        .groupby(["store_num", "week_id"])
        .sum(),
        join="inner",
    )
)

asset_counts_rule = gppd.add_constrs(
    model,
    LHS,
    GRB.LESS_EQUAL,
    M * RHS,
    name="asset_counts_rule",
)
# print("Debugging starts here")
# # print(asset_counts_rule.apply(model.getRow))
# # print(RHS)
# print("LHS and RHS:")
# idx = pd.IndexSlice
# print(LHS.loc[idx[456, :]])
# print(RHS.loc[idx[456, :]])
# print("Week 38 Resource Availability")
# print(df_asset_counts_U_ikn.xs((456, "2023_38"), level=("store_num", "week_id")))

# Constraint 2: Max one asset count per week for each resource - sum_k (u_ikn)<= 1 for all i and n
one_asset_count_per_week = gppd.add_constrs(
    model,
    df_asset_counts_U_ikn["u"].groupby(["resource_id", "week_id"]).sum(),
    GRB.LESS_EQUAL,
    1,
    name="one_store_per_week",
)

# Constraint 3: Ensure that if consultant is assigned a changeover he/she can't do asset counts on that week - u_ik'n <= (1-x_ik) for all i, k, k' and n=T_k
df1 = df_asset_counts_U_ikn.reset_index()
df2 = df_consultant_store_X_ik.reset_index().drop(
    columns=[
        "string_id",
        "changeover_date",
        "c445_yr_num",
        "c445_wk_num",
        "travel_distance_km",
    ],
)
df_joined = pd.merge(
    df1,
    df2,
    how="inner",
    left_on=["resource_id", "week_id"],
    right_on=[
        "resource_id",
        "changeover_week_id",
    ],
).gppd.add_constrs(model, "u<=1-x", name="assets_or_changeover_per_week")

# Constratint 4: Each store must get one consultant for the changeover - \sum x_{ik} == 1 for each k
store_coverage = gppd.add_constrs(
    model,
    df_consultant_store_X_ik["x"]
    .groupby(
        "store_num",
    )
    .sum(),
    GRB.EQUAL,
    1,
    name="store_coverage",
)

# Constratint 5: Ensures that each consultant can't be assigned to more than one store changeover on a given chaneover week
# sum_k(x_ik) <= 1 for all i
total_changeovers = df_consultant_store_X_ik.groupby(
    ["changeover_week_id", "resource_id"],
)["x"].sum()
allocate_once = gppd.add_constrs(
    model,
    total_changeovers,
    GRB.LESS_EQUAL,
    1,
    name="allocate_once",
)

# Constraint 6 - Changeover-Asset Counts: Making sure if consultant is assigned to the changeover is also assigned to the asset counts
# x_ik <= sum_n (u_ikn) for all i and k

LHS, RHS = df_consultant_store_X_ik["x"].align(
    df_asset_counts_U_ikn["u"].groupby(["resource_id", "store_num"]).sum(),
    join="inner",
)
constulant_changeover_coverage = gppd.add_constrs(
    model,
    LHS,
    GRB.LESS_EQUAL,
    RHS,
    name="consultant_changeover_coverage",
)

# Constraint 7 -Count the number of consultants assigned to the string
# x_ik <= y_is for all i, k, s

df_joined = pd.merge(
    df_consultant_string.reset_index(),
    df_consultant_store_X_ik.reset_index(),
    how="right",
    on=["resource_id", "string_id"],
)
df = df_joined[["resource_id", "string_id", "store_num"]].drop_duplicates(
    subset=["resource_id", "string_id", "store_num"],
)

constraint_index = pd.Series(
    index=pd.MultiIndex.from_frame(
        df,
        names=["resource_id", "string_id", "store_num"],
    ),
)

LHS, _ = df_consultant_store_X_ik["x"].align(
    constraint_index,
    join="inner",
)

RHS, _ = y.align(constraint_index, join="inner")
LHS = LHS.reorder_levels(constraint_index.index.names)
RHS = RHS.reorder_levels(constraint_index.index.names)

filter = LHS.notna() & RHS.notna()
LHS, RHS = LHS[filter], RHS[filter]

consultant_to_string_assignment = gppd.add_constrs(
    model,
    LHS,
    GRB.LESS_EQUAL,
    RHS,
    name="consultant_to_string_assignment",
)

# ************************************************************************************************************************
#                                                5. Auxilary Constraints - For Objective Function
# ************************************************************************************************************************

# Constraint 8 - Objective: Total number of additional resource-weeks required for asset counts across all stores and weeks
model.addConstr(z.sum() == w, "total_additional_resources")

# Constraint 9 - Objective: Assignment indicator variable for distance
LHS, RHS = (
    df_asset_counts_U_ikn["u"]
    .groupby(["resource_id", "store_num"])
    .sum()
    .align(
        t,
        join="inner",
    )
)
aux_distance = gppd.add_constrs(
    model,
    LHS,
    GRB.LESS_EQUAL,
    RHS,
    name="aux_distance",
)
total_distance = (df_travel_distance_r_ik["travel_distance_km"] * t).sum()

# Constraint 10 - Objective: Total Workload/Fairness
total_workload = gppd.add_constrs(
    model,
    df_asset_counts_U_ikn["u"].groupby("resource_id").sum(),
    GRB.EQUAL,
    total_assignments,
    name="total_workload",
)

min_constr = model.addGenConstrMin(
    minAssignment,
    total_assignments,
    name="minAssignment",
)
max_constr = model.addGenConstrMax(
    maxAssignment,
    total_assignments,
    name="maxAssignment",
)

# ******************************************************************************************************
#                                   6. Multiple Objective Functions
# *******************************************************************************************************
# total_assignments = u.groupby(['resource_id', 'store_num']).sum()
# total_distance = (
#     df_travel_distance_r_ik["travel_distance_km"] * df_asset_counts_U_ikn["u"].sum()
# ).sum()
# print(total_distance)
# model.setObjective(total_distance, sense=GRB.MINIMIZE)
# model.setObjective(0, sense=GRB.MINIMIZE)

model.setObjectiveN(
    y.sum(),
    index=0,
    priority=4,
    name="Total_Consultants_Assigned_to_String",
)

model.setObjectiveN(
    w,
    index=1,
    priority=3,
    name="Extra_Resources",
)
model.setObjectiveN(
    total_distance,
    index=2,
    priority=2,
    name="Total_Distance",
)
model.setObjectiveN(
    maxAssignment - minAssignment,
    index=3,
    priority=1,
    name="Fairness",
)

# print(model.ModelSense)
# model.ModelSense = GRB.MINIMIZE
model.optimize()
# model.write('model/prescriptive/workforce.lp')

if model.status == gp.GRB.OPTIMAL:
    print("Model solved to optimality.")
elif model.status == gp.GRB.INFEASIBLE:
    print("Model is infeasible.")
elif model.status == gp.GRB.UNBOUNDED:
    print("Model is unbounded.")
else:
    print("Unknown status.")

# COMMAND ----------

# model.getVars()
# print('-'*50)
# print('')
print("********************** OUTPUT REPORTS & CHARTS******************")
print("")
print("")
print("************************** Changeover Stats ********************")
print(
    "Total Changeovers: ",
    df_consultant_store_X_ik["x"]
    .gppd.X.to_frame()
    .query("x > 0")
    .reset_index()["store_num"]
    .nunique(),
)
print(
    "Total Resources in Changeovers: ",
    df_consultant_store_X_ik["x"]
    .gppd.X.to_frame()
    .query("x > 0")
    .reset_index()["resource_id"]
    .nunique(),
)
print("*********************String Stats*******************************")
print(
    "Total Strings:",
    y.gppd.X.to_frame().query("y > 0").reset_index()["string_id"].nunique(),
)
print(
    "Total Resources in Strings:",
    y.gppd.X.to_frame().query("y > 0").reset_index()["resource_id"].nunique(),
)

print("***************** Num of Consultant Per String *********************")
# shift_table = y.gppd.X.unstack(0).fillna("-").replace({0.0: "-", 1.0: "Y"})
# print(shift_table)

df = y.gppd.X.to_frame().query("y == 1.0").reset_index()
plt.bar(
    df["string_id"].unique(),
    df.groupby(
        ["string_id"],
    )["resource_id"].count(),
    align="center",
)
plt.title("Consultants Per String", fontsize=14)
plt.xticks(df["string_id"].unique())
# plt.yticks(df.groupby(['resource_id'])['store_num'].count())
plt.ylim(0, 5)
plt.ylabel("Num. of Consultants", fontsize=14)
plt.xlabel("Strings", fontsize=14)
plt.show()

print("*********************Asset Counts Stats**************************")
print(
    "Total Resources: ",
    df_asset_counts_U_ikn["u"]
    .gppd.X.to_frame()
    .query("u > 0")
    .reset_index()["resource_id"]
    .nunique(),
)
print(
    "Total Stores: ",
    df_asset_counts_U_ikn["u"]
    .gppd.X.to_frame()
    .query("u > 0")
    .reset_index()["store_num"]
    .nunique(),
)

print("*****************Asset Counts Shortage***************************")
print(
    "Assets Shortage: ",
    z.gppd.X.to_frame().query("z > 0").reset_index()["store_num"].nunique(),
)
print(
    "Store-Week: \n",
    z.gppd.X.to_frame().query("z > 0").reset_index(),
)
# print("*****************Changeover Shortage*****************************")
# print(
#     "Changeover Shortage: ",
#     df_extra["Extra"]
#     .gppd.X.to_frame()
#     .query("Extra > 0")
#     .reset_index()["store_num"]
#     .nunique(),
# )
# print(df_extra["Extra"].gppd.X.to_frame().query("Extra > 0").reset_index())
print("")
if model.Status == GRB.OPTIMAL:
    print("Total Extra Personnel Needed (Shortage):", w.X)
print("")
print("")

print("*****************************Changeover Workload***************************")
df = (
    df_consultant_store_X_ik["x"]
    .gppd.X.to_frame()
    .query(
        "x == 1.0",
    )
    .reset_index()
)
plt.bar(
    df["resource_id"].unique(),
    df.groupby(
        ["resource_id"],
    )["store_num"].count(),
    align="center",
)
plt.title("Store Changeover Workload", fontsize=14)
plt.xticks(df["resource_id"].unique())
# plt.yticks(df.groupby(['resource_id'])['store_num'].count())
plt.ylim(0, 25)
plt.xlabel("Consultants", fontsize=14)
plt.ylabel("Number of Stores", fontsize=14)
plt.show()

print("\n")
print(
    "*****************************Asset Counts Workload**************************",
)
# print (total_assignments)
df = total_assignments.gppd.X.to_frame().reset_index()
plt.bar(
    df["resource_id"],
    df["total_assignments"],
    align="center",
)
plt.title("Asset Counts Workload", fontsize=12)
plt.xticks(df["resource_id"].unique())
# plt.yticks(df.groupby(['resource_id'])['store_num'].count())
plt.ylim(0, 50)
plt.xlabel("Consultants and Analysts", fontsize=12)
plt.ylabel("Number of Store-Weeks", fontsize=12)
plt.show()
print(
    "*******************************Heatmap**************************************",
)
df_asset_counts_U_ikn = (
    df_asset_counts_U_ikn.assign(
        u_val=df_asset_counts_U_ikn["u"].gppd.X,
    )
    .reset_index()
    .drop(columns=["u"])
    .query("u_val == 1.0")
)

df_asset_counts_U_ikn.loc[:, "task"] = "Asset Counts"

df_changeover_X_ik = (
    df_consultant_store_X_ik.assign(
        x_val=df_consultant_store_X_ik["x"].gppd.X,
    )
    .reset_index()
    .drop(columns=["x"])
    .query("x_val == 1.0")
)

df_changeover_X_ik = df_changeover_X_ik.rename(
    columns={"changeover_week_id": "week_id"},
)
print(
    "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Debugging Starts--------------------------------------------",
)
print(df_asset_counts_U_ikn.info())
print(df_changeover_X_ik.info())
# print(
#     df_extra.assign(extra=df_extra["Extra"].gppd.X)
#     .reset_index()
#     .drop(columns=["Extra"])
#     .query("extra == 1.0")
# )
# print(z.gppd.X.to_frame().reset_index().query("z == 1.0"))

df_changeover_X_ik.loc[:, "task"] = "Changeover"

cols = [
    "resource_id",
    "store_num",
    "string_id",
    "week_id",
    "changeover_date",
    "travel_distance_km",
    "c445_yr_num",
    "c445_wk_num",
    "task",
]

role_cols = ["resource_is_consultant", "resource_is_analyst"]

df = pd.concat(
    [
        df_asset_counts_U_ikn[cols + role_cols],
        df_changeover_X_ik[cols],
    ],
)

df["viz_color"] = df["string_id"].astype("category").factorize()[0]
df["store_num"] = df["store_num"].astype("str", errors="ignore")
df["store_num"] = df["store_num"].str.pad(4, "left", "0")
df["string_id"] = df["string_id"].astype("str", errors="ignore")
df["resource_is_consultant"] = (
    df["resource_is_consultant"]
    .fillna(
        1,
    )
    .astype(int)
)
df["resource_is_analyst"] = df["resource_is_analyst"].fillna(0).astype(int)
# df["role"] = df["resource_is_consultant"].apply(lambda x: 'Consultant' if x == 1 else 0)
df["role"] = df["resource_is_analyst"].apply(
    lambda x: "Analyst" if x == 1 else "Consultant",
)
print(df.info())
# print("\n")
# print("before casting")
# print(df_string_raw.info())
# print(df_string_raw[df_string_raw["store_num"] == 840])

df_string_raw["string"] = df_string_raw["string"].astype(
    "str",
    errors="ignore",
)
# df_string_raw["store_num"] = (
#     df_string_raw["store_num"]
#     .astype(
#         str,
#         errors="ignore",
#     )
#     .apply(lambda x: x.rstrip(".0"))
# )
# df_string_raw["store_num"] = df_string_raw["store_num"].str.pad(
#     4,
#     "left",
#     "0",
# )
df_string_raw["store_num"] = (
    df_string_raw["store_num"].astype(int).astype(str).str.zfill(4)
)
# print("After")
df_string_raw = df_string_raw.rename(
    {"string": "string_id"},
    axis="columns",
)
# print("After the casting")
# print(df_string_raw.info())
# print("\n")
# print(df_string_raw[df_string_raw["string_id"] == "23K"])
# print('Before Join: ')
# print(
#     'Number of Total Asset Counts in Model: ' +
#     str(len(df.query('task == "Asset Counts"'))),
# )
# print(
#     'Number of Changeovers in Model: ' +
#     str(len(df.query('task == "Changeover"'))),
# )

df_joined1 = pd.merge(
    # pd.merge(
    df,
    df_store_dim[
        [
            "store_num",
            "store_nm",
            "city_nm",
            "language_cd",
        ]
    ],
    how="left",
    on="store_num",
)

# print(df_joined1.info())
# print('After Join 1: ')
# print(
#     'Number of Total Asset Counts in Model:' +
#     str(len(df_joined1.query('task == "Asset Counts"'))),
# )
# print(
#     'Number of Changeovers in Model' +
#     str(len(df_joined1.query('task == "Changeover"'))),
# )

df_joined2 = pd.merge(
    df_joined1,
    df_string_raw[
        [
            "store_num",
            "string_id",
            "string_order",
            "date_in_store",
            "asset_type",
            "type",
            "sqft",
        ]
    ],
    how="left",
    on=["store_num", "string_id"],
)

df_joined2["date_in_store"] = pd.to_datetime(
    df_joined2["date_in_store"],
    errors="coerce",
)
df_joined2["last_co_date"] = df_joined2["date_in_store"] - pd.Timedelta(days=1)
df_joined2["last_co_date"] = df_joined2["last_co_date"].dt.date

# print(df_joined2.info())
# print(df_joined2.head())
# print(df_string_raw.info())

# df_joined ["week_rank"] = df_joined ['week_id'].rank(method = 'min').astype('int')
# print(df_joined.head())
# print(df_joined.info())
# print(df_joined.duplicated().sum())
# print (df_joined.isnull().sum())
# print (df_joined[df_joined['store_nm'].isnull()])

# print(df_joined2["c445_yr_num"].unique())

# print('After Join 2')
# print(
#     'Number of Total Asset Counts in Model: ' +
#     str(len(df_joined2.query('task == "Asset Counts"'))),
# )
# print(
#     'Number of Changeovers in Model: ' +
#     str(len(df_joined2.query('task == "Changeover"'))),
# )

# print(df.info())
# print(df.shape)
print(
    "Total Changeovers - All Years: ",
    df[df["task"] == "Changeover"].shape[0],
)
print(
    "Total Assets-All Years: ",
    df[df["task"] == "Asset Counts"].shape[0],
)
print("\n")
df = df_joined2.query("c445_yr_num >= 2023").copy()
print(
    "Total Changeovers - 2023: ",
    df[df["task"] == "Changeover"].shape[0],
)
print(
    "Total Assets - 2023: ",
    df[df["task"] == "Asset Counts"].shape[0],
)

print(df.info())
# print(len(df["store_num"].unique()))
# df1 = df.groupby('store_num').size().reset_index()
# print(df1.columns.tolist())
# df1.loc[:,"store_num"] = df1.loc[:, "store_num"].astype('int')
# df2 = df_asset_counts_U_ikn.drop_duplicates("store_num")[["store_num", "num_resource_weeks_needed"]]
# df2.loc[:,"store_num"] = df2.loc[:, "store_num"].astype('int')
# df1.merge (df2, "inner", on = ["store_num"]).to_csv('qa.csv')
# df.query("c4c445_wk_num >= 45").to_csv ()

# df = df_joined2.query('c445_yr_num >= 2023').copy()

# print("Here 2023 " + str(df['c445_wk_num'].isna().sum()))

df = df.sort_values(
    by=["c445_yr_num", "c445_wk_num"],
    ascending=[True, True],
)

df["push_button"] = "Push Button"

df = df.fillna("NA")
# print('FINAL TABLE FOR HEATMAP')
# print(df.query('store_num == "0445"')[["resource_id", "store_num", "string_id", "week_id", "c445_yr_num", "c445_wk_num", "task"]])
# print(df.query('store_num == "0244"')[["resource_id", "store_num", "string_id", "week_id", "c445_yr_num", "c445_wk_num", "task"]])
# print(df.query('store_num == "0481"')[["resource_id", "store_num", "string_id", "week_id", "c445_yr_num", "c445_wk_num", "task"]])
# print(df.query('resource_id == "Erin D"')[["resource_id", "store_num", "string_id", "week_id", "c445_yr_num", "c445_wk_num", "task"]])
# print(df.query('resource_id == "Lynn M" and week_id == "2023_01"')[["resource_id", "store_num", "string_id", "week_id", "c445_yr_num", "c445_wk_num", "task"]])

# print(df_joined.duplicated().sum())
# print (df_joined.isnull().sum())
# df_joined.to_csv('output.csv')
week_count = df["week_id"].nunique()
resource_count = df["resource_id"].nunique()
# print(week_count)

# colorscale = [
#     (0, '#FFE5AD'),  # orange
#     (1, '#BBCF8C'),  # green
# ]

# Define custom order of resource IDs
custom_order = [
    "Ian J",
    "Lynn M",
    "Erin D",
    "Adam S",
    "Genna A",
    "Tim M",
    "Mira G",
    "Ray B",
    "Dercio R",
    "Gigi K",
    "Rami K",
    "Shafik J",
    "Gifford M",
    "Mihir P",
    "James M",
]

margin = dict(t=150, b=80, l=385, r=50)

fig = go.Figure(
    data=go.Heatmap(
        y=[
            df["c445_yr_num"].astype("int"),
            df["c445_wk_num"].astype("int"),
        ],
        x=df["resource_id"],
        z=df["viz_color"],
        zauto=True,
        # zmin=-0.0,
        # zmax=1.0,
        # colorscale=colorscale,
        showscale=False,
        text=(
            df.apply(
                lambda r: f'{r["store_num"]:}<br>'
                f'<sub style = "font-size: 12pt">{r["city_nm"]:}</sub><br>'
                f'<sub style = "font-size: 11pt">{r["push_button"]:}</sub><br>'
                f'<sub style = "font-size: 11pt">{r["type"]:}</sub>',
                axis=1,
            )
        ),
        texttemplate="%{text}",
        textfont={
            "size": 12,
            "color": "#FFFFFF",
        },
        customdata=df,
        hovertemplate=(
            " <b>Week</b>: %{y}  <br>"
            " <b>Week_ID</b>: %{customdata[3]}  <br>"
            " <br>"
            " <b>Task</b>: %{customdata[8]} <br>"
            " <br>"
            " <b>Resource</b>: %{customdata[0]} <br>"
            " . <i>Role:</i>... %{customdata[12]} <br>"
            " . <i>Travel Distance (km):</i>... %{customdata[5]} <br>"
            " <br>"
            " <b>Store Name</b>: %{customdata[13]} <br>"
            " . <i>Store Num:</i>.............. %{customdata[1]} <br>"
            " . <i>Store Size (sqft):</i>.......%{customdata[20]} <br>"
            " . <i>Language:</i>............... %{customdata[15]} <br>"
            " . <i>Type of Listing:</i>.........%{customdata[19]} <br>"
            " . <i>Changeover Date:</i>.........%{customdata[4]} <br>"
            " . <i>Last Changeover Date:</i>....%{customdata[21]} <br>"
            " <br>"
            " <b>String ID</b>:..... %{customdata[2]} <br>"
            " <b>String Order</b>:.. %{customdata[16]} <br>"
            # " <br>"
            # " <b>Week_ID</b>: %{customdata[3]}  <br>"
            "<extra></extra>"
        ),
        hoverlabel={"font_family": "monospace"},
        xgap=3,
        ygap=3,
    ),
    layout=go.Layout(
        title=("<b>Assignment Schedule (2023-2024-2025) </b><br>"),
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=margin,
        width=margin["l"] + margin["r"] + resource_count * 100,
        height=margin["t"] + margin["b"] + week_count * 70,  # 2 * 52* 40
        xaxis=go.layout.XAxis(
            title=go.layout.xaxis.Title(
                text="<b>Resources</b>",
                standoff=20,
            ),
            side="top",
            showdividers=False,
            fixedrange=True,
            tickfont=dict(size=14),
            categoryorder="array",
            categoryarray=custom_order,
        ),
        yaxis=go.layout.YAxis(
            title="<b>Weeks</b>",
            ticks="outside",
            tickcolor="white",
            ticklen=10,
            tickfont=dict(size=14),
            fixedrange=True,
            autorange="reversed",
        ),
    ),
)

# # Add overlays on top of inactive products
# annotation_style = {
#     'bgcolor': '#FFF',
#     'bordercolor': '#CCC',
#     'borderpad': 0,
#     'borderwidth': 1,
#     'height': 70 - 5,
#     'opacity': 0.8,
#     'showarrow': False,
#     'text': '✕',
#     'font': {
#         'color': '#2A3F5F',
#     },
#     'width': 90- 5,
#     'xshift': -1,
#     'yshift': 1,
# }

# create a mask for "Changeover" rows
changeover_mask = df["task"] == "Changeover"

# add annotations for "Changeover" cells
annotations = []
for i, row in df[changeover_mask].iterrows():
    annotations.append(
        dict(
            x=row["resource_id"],
            y=row[["c445_yr_num", "c445_wk_num"]].tolist(),
            text="C",
            opacity=0.35,
            showarrow=False,
            font=dict(color="#2A3F5F", size=16),
            borderpad=0,
            bgcolor="#FFF",
            bordercolor="#CCC",
            borderwidth=1,
            xshift=-1,
            yshift=1,
        ),
    )

fig.update_layout(
    annotations=annotations,
)

fig.show()
