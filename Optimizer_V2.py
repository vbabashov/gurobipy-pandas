# Databricks notebook source
# MAGIC %md
# MAGIC # Dealer Changeover PoC: Prescriptive Analytics Approach

# COMMAND ----------

# MAGIC %md
# MAGIC Authors: Vusal Babashov & Evan Sinukoff

# COMMAND ----------

# MAGIC %md
# MAGIC ## Business Problem

# COMMAND ----------

# MAGIC %md
# MAGIC Given the string and store information, allocate resources efficiently to strings and stores. 

# COMMAND ----------

# MAGIC %md
# MAGIC ![string.png](attachment:string.png)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model

# COMMAND ----------

# MAGIC %md
# MAGIC  - Mixed Integer Programming
# MAGIC  - Python, Gurobi

# COMMAND ----------

# MAGIC %md
# MAGIC Indices and Sets

# COMMAND ----------

# MAGIC %md
# MAGIC - $i \in I$:  Index and set of consultant and analysts
# MAGIC - $k \in K$:  Index and set of stores
# MAGIC - $s \in S$:  Index and set of strings
# MAGIC - $n \in N$:  Index and set of weeks

# COMMAND ----------

# MAGIC %md
# MAGIC - $I(k)$:  Set of people who can work in store $k$ (i.e., language, availability)
# MAGIC - $I(s)$:  Set of people who can do all stores in the string $s$ (i.e., language and avaiability on $T_k$)
# MAGIC - $I(n)$:  Set of people who are avaialable on week $n$ (i.e., vacations and blackout weeks)
# MAGIC - $I_c$:   Set of consultants
# MAGIC - $K(s)$:  Set of stores that belong string $s$ 
# MAGIC - $N(k)$:  Set of weeks in which store $k$ asset counts can be done (i.e., between 4 and 8 weeks prior to changeover date)
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Parameters

# COMMAND ----------

# MAGIC %md
# MAGIC - $T_{k}$ $\in\mathbb{N}$: Changeover week at a store $k\in K$
# MAGIC - $D_{k}$ $\in\mathbb{N}$ : Number of person-weeks required for asset listing at store $k \in K$  (i.e., one person for every 10,000 sf)
# MAGIC - $r_{ik}$: Distance between home province for consultant/analyst $i$ and store $k$ 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Decision Variables:

# COMMAND ----------

# MAGIC %md
# MAGIC - $y_{is} \in \{0,1\}$:   Equals 1 if consultant $i\in I$ is assigned to string $s \in S$, 0 otherwise
# MAGIC - $x_{ik} \in \{0,1\}$:   Equals 1 if consultant $i\in I_c$ is assigned to changeover at store $k \in K$, 0 otherwise
# MAGIC <!-- - $Extra_{ik} \in \{0,1\}$:   Equals 1 if consultant $i\in I_c$ is assigned to changeover at store $k \in K$, 0 otherwise -->
# MAGIC - $u_{ikn} \in \{0,1\}$:  Equals 1 if consultant/analysts $i\in I$, is assigned to asset counts at store $k \in K$ on week $n \in N$, 0 otherwise
# MAGIC - $z_{kn} \in\mathbb{N}$: Number of extra staff required to satisfy the requirements of store $k \in K$ asset counts on week $n \in N$
# MAGIC - $w$ : Auxilary variable to track the number of extra staff
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Objective Function

# COMMAND ----------

# MAGIC %md
# MAGIC \begin{equation}
# MAGIC \text{Min} \quad \sum_{i \in I(k)} \sum_{s \in S} y_{is}
# MAGIC \end{equation}

# COMMAND ----------

# MAGIC %md
# MAGIC Third Party Staff

# COMMAND ----------

# MAGIC %md
# MAGIC \begin{equation}
# MAGIC \text{Min} \quad w 
# MAGIC \end{equation}

# COMMAND ----------

# MAGIC %md
# MAGIC Total Distance

# COMMAND ----------

# MAGIC %md
# MAGIC \begin{equation}
# MAGIC \text{Min} \quad \sum_{i \in I(k)} \sum_{k}  \left (r_{ik} t_{ik}\right)
# MAGIC \end{equation}

# COMMAND ----------

# MAGIC %md
# MAGIC Fairness in Workload/Assignments

# COMMAND ----------

# MAGIC %md
# MAGIC \begin{equation}
# MAGIC \text{Min} \quad \{\text{maxAssign} - \text{minAssign} \}
# MAGIC \end{equation}

# COMMAND ----------

# MAGIC %md
# MAGIC ### Constraints

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1a. Staffing Requirements: Person-Week Requirement For Store Asset Counts

# COMMAND ----------

# MAGIC %md
# MAGIC \begin{equation}
# MAGIC \sum_{n \in N(k) } \left( z_{kn}  + \sum_{i \in I(k) \cap I(n) } u_{ikn} \right)  = D_{k} \quad \forall\; k \in K
# MAGIC \end{equation}

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1b. Ensure that analyst is not there alone on a given asset count week and store. e.g., There is always a consultant 

# COMMAND ----------

# MAGIC %md
# MAGIC \begin{equation}
# MAGIC \sum_{i \in I(k) \cap I(n) } u_{ikn}   <= M* \sum_{i \in I(k) , I_c } u_{ikn}\quad \forall\; k \in K, n \in N(k)
# MAGIC \end{equation}

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2. Max One-Store Asset Count Per Week

# COMMAND ----------

# MAGIC %md
# MAGIC \begin{equation}
# MAGIC \sum_{k \in K} u_{ikn}  <= 1 \quad \forall\; i \in I (k), n \in N(k), 
# MAGIC \end{equation}

# COMMAND ----------

# MAGIC %md
# MAGIC #### 3. Ensure either Asset Counts or Changeover on the Same Week

# COMMAND ----------

# MAGIC %md
# MAGIC \begin{equation}
# MAGIC u_{ik'n} \le (1 - x_{ik})  \quad \forall\; i \in I, k,k' \in K\,, k' \ne k, n = T_k
# MAGIC \end{equation}

# COMMAND ----------

# MAGIC %md
# MAGIC #### 4. One Consultant Per Each Store Changeover

# COMMAND ----------

# MAGIC %md
# MAGIC \begin{equation}
# MAGIC \sum_{i \in I(k) \cap I_c \cap I(T_k) } x_{ik} = 1 \quad \forall\; k \in K, \;
# MAGIC \end{equation}

# COMMAND ----------

# MAGIC %md
# MAGIC #### 5. Max One Store per Consultant on Each Changeover Week

# COMMAND ----------

# MAGIC %md
# MAGIC \begin{equation}
# MAGIC \sum_{k \in K(T_k)} x_{ik} <= 1 \quad \forall\; i \in I(k) \cap I_c \cap I(T_k) , \;
# MAGIC \end{equation}

# COMMAND ----------

# MAGIC %md
# MAGIC Applies to stores with the same changeover weeks 

# COMMAND ----------

# MAGIC %md
# MAGIC #### 6. Changeover - Asset Count Relationship

# COMMAND ----------

# MAGIC %md
# MAGIC \begin{equation}
# MAGIC x_{ik} <= \sum_{n \in N(k)} u_{ikn} \quad \forall\; i \in I(k), k \in K
# MAGIC \end{equation}

# COMMAND ----------

# MAGIC %md
# MAGIC #### 7.Count the number of consultants assigned to the string

# COMMAND ----------

# MAGIC %md
# MAGIC \begin{equation}
# MAGIC x_{ik} <= y_{is} \quad \forall\; i \in I(k), k \in K(s), s \in S
# MAGIC \end{equation}

# COMMAND ----------

# MAGIC %md
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC #### 8. Auxilary Constraint: Total Additional Resources Needed

# COMMAND ----------

# MAGIC %md
# MAGIC \begin{equation}
# MAGIC \sum_{k}\sum_{n \in N(k)}  z_{kn}=  w
# MAGIC \end{equation}

# COMMAND ----------

# MAGIC %md
# MAGIC #### 9.Auxilary Constraint: Assignment indicator for distance 

# COMMAND ----------

# MAGIC %md
# MAGIC \begin{equation}
# MAGIC  \sum_{n\in N(k)} u_{ikn} <= t_{ik}   \quad \forall\; i \in I (k), k \in K
# MAGIC \end{equation}

# COMMAND ----------

# MAGIC %md
# MAGIC #### 10. Auxilary Constraint: Fairness/Total Workload

# COMMAND ----------

# MAGIC %md
# MAGIC \begin{equation}
# MAGIC \sum_{k \in K} \sum_{n\in N(k)} u_{ikn}  = \text{totAssignments(i)} \quad \forall\; i \in I (k)
# MAGIC \end{equation}

# COMMAND ----------

# MAGIC %md
# MAGIC \begin{equation}
# MAGIC \text{maxNumAssign} = \text{Max} \{\text{totAssignments}(i):  \quad \forall\; i \in I \}
# MAGIC \end{equation}
# MAGIC \begin{equation}
# MAGIC \text{minNumAssign} = \text{Min} \{\text{totAssignments}(i):  \quad \forall\; i \in I \}  
# MAGIC \end{equation}

# COMMAND ----------

# MAGIC %md
# MAGIC #### Demo Results

# COMMAND ----------

# MAGIC %md
# MAGIC ![string_workload.png](attachment:string_workload.png)

# COMMAND ----------

# MAGIC %md
# MAGIC ![changeover.png](attachment:changeover.png)

# COMMAND ----------

# MAGIC %md
# MAGIC ![assets.png](attachment:assets.png)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### See Heatmap For Detailed Assignments

# COMMAND ----------

# MAGIC %md
# MAGIC #### Future Work:

# COMMAND ----------

# MAGIC %md
# MAGIC  - Vacations
# MAGIC  - Duplicate store records in a given string with different changeover dates
# MAGIC  - Training, Retiring Dealers
# MAGIC  - Changeover as a decision variable?
# MAGIC  - Dealers running two stores at the same time?
# MAGIC  - Heatmap improvements

# COMMAND ----------

# MAGIC %md
# MAGIC #### Next Steps:

# COMMAND ----------

# MAGIC %md
# MAGIC  - Feedback on quality of solution
# MAGIC  - Go or No-go Decision

# COMMAND ----------

# MAGIC %md
# MAGIC End..
