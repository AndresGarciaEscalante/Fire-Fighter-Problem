from statistics import mode
import pandas as pd
import random
import os
import math
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Provides the methods to create and solve the firefighter problem
class FFP:

  # Constructor
  #   fileName = The name of the file that contains the FFP instance
  def __init__(self, fileName):
    file = open(fileName, "r")    
    text = file.read()    
    tokens = text.split()
    seed = int(tokens.pop(0))
    self.n = int(tokens.pop(0))
    model = int(tokens.pop(0))  
    int(tokens.pop(0)) # Ignored
    # self.state contains the state of each node
    #    -1 On fire
    #     0 Available for analysis
    #     1 Protected
    self.state = [0] * self.n
    nbBurning = int(tokens.pop(0))
    for i in range(nbBurning):
      b = int(tokens.pop(0))
      self.state[b] = -1      
    self.graph = []    
    for i in range(self.n):
      self.graph.append([0] * self.n);
    while tokens:
      x = int(tokens.pop(0))
      y = int(tokens.pop(0))
      self.graph[x][y] = 1
      self.graph[y][x] = 1    

  # Solves the FFP by using a given method and a number of firefighters
  #   method = Either a string with the name of one available heuristic or an object of class HyperHeuristic
  #   nbFighters = The number of available firefighters per turn
  #   debug = A flag to indicate if debugging messages are shown or not
  def solve(self, method, nbFighters, debug = False):
    spreading = True
    if (debug):
      print("Initial state:" + str(self.state))    
    t = 0
    while (spreading):
      if (debug):
        print("Features")
        print("")
        print("Graph density: %1.4f" % (self.getFeature("EDGE_DENSITY")))
        print("Average degree: %1.4f" % (self.getFeature("AVG_DEGREE")))
        print("Burning nodes: %1.4f" % self.getFeature("BURNING_NODES"))
        print("Burning edges: %1.4f" % self.getFeature("BURNING_EDGES"))
        print("Nodes in danger: %1.4f" % self.getFeature("NODES_IN_DANGER"))
      # It protects the nodes (based on the number of available firefighters)
      for i in range(nbFighters):
        heuristic = method
        if (isinstance(method, HyperHeuristic)):
          heuristic = method.nextHeuristic(self)
        node = self.__nextNode(heuristic)
        if (node >= 0):
          # The node is protected   
          self.state[node] = 1
          # The node is disconnected from the rest of the graph
          for j in range(len(self.graph[node])):
            self.graph[node][j] = 0
            self.graph[j][node] = 0
          if (debug):
            print("\tt" + str(t) + ": A firefighter protects node " + str(node))            
      # It spreads the fire among the unprotected nodes
      spreading = False 
      state = self.state.copy()
      for i in range(len(state)):
        # If the node is on fire, the fire propagates among its neighbors
        if (state[i] == -1): 
          for j in range(len(self.graph[i])):
            if (self.graph[i][j] == 1 and state[j] == 0):
              spreading = True
              # The neighbor is also on fire
              self.state[j] = -1
              # The edge between the nodes is removed (it will no longer be used)
              self.graph[i][j] = 0
              self.graph[j][i] = 0
              if (debug):
                print("\tt" + str(t) + ": Fire spreads to node " + str(j))     
      t = t + 1
      if (debug):
        print("---------------")
    if (debug):    
      print("Final state: " + str(self.state))
      print("Solution evaluation: " + str(self.getFeature("BURNING_NODES")))
    return self.getFeature("BURNING_NODES")

  # Selects the next node to protect by a firefighter
  #   heuristic = A string with the name of one available heuristic
  def __nextNode(self, heuristic):
    index  = -1
    best = -1
    for i in range(len(self.state)):
      if (self.state[i] == 0):
        index = i        
        break
    value = -1
    for i in range(len(self.state)):
      if (self.state[i] == 0):
        if (heuristic == "LDEG"):
          # It prefers the node with the largest degree, but it only considers
          # the nodes directly connected to a node on fire
          for j in range(len(self.graph[i])):
            if (self.graph[i][j] == 1 and self.state[j] == -1):
              value = sum(self.graph[i])              
              break
        elif (heuristic == "GDEG"):        
          value = sum(self.graph[i])          
        else:
          print("=====================")
          print("Critical error at FFP.__nextNode.")
          print("Heuristic " + heuristic + " is not recognized by the system.")          
          print("The system will halt.")
          print("=====================")
          exit(0)
      if (value > best):
        best = value
        index = i
    return index

  # Returns the value of the feature provided as argument
  #   feature = A string with the name of one available feature
  def getFeature(self, feature):
    f = 0
    if (feature == "EDGE_DENSITY"):
      n = len(self.graph)      
      for i in range(len(self.graph)):
        f = f + sum(self.graph[i])
      f = f / (n * (n - 1))
    elif (feature == "AVG_DEGREE"):
      n = len(self.graph) 
      count = 0
      for i in range(len(self.state)):
        if (self.state[i] == 0):
          f += sum(self.graph[i])
          count += 1
      if (count > 0):
        f /= count
        f /= (n - 1)
      else:
        f = 0
    elif (feature == "BURNING_NODES"):
      for i in range(len(self.state)):
        if (self.state[i] == -1):
          f += 1
      f = f / len(self.state)
    elif (feature == "BURNING_EDGES"):
      n = len(self.graph) 
      for i in range(len(self.graph)):
        for j in range(len(self.graph[i])):
          if (self.state[i] == -1 and self.graph[i][j] == 1):
            f += 1
      f = f / (n * (n - 1))    
    elif  (feature == "NODES_IN_DANGER"):
      for j in range(len(self.state)):
        for i in range(len(self.state)):
          if (self.state[i] == -1 and self.graph[i][j] == 1):
            f += 1
            break
      f /= len(self.state)
    else:      
      print("=====================")
      print("Critical error at FFP._getFeature.")
      print("Feature " + feature + " is not recognized by the system.")          
      print("The system will halt.")
      print("=====================")
      exit(0)
    return f

  # Returns the string representation of this problem
  def __str__(self):
    text = "n = " + str(self.n) + "\n"
    text += "state = " + str(self.state) + "\n"
    for i in range(self.n):
      for j in range(self.n):
        if (self.graph[i][j] == 1 and i < j):
          text += "\t" + str(i) + " - " + str(j) + "\n"
    return text

# Provides the methods to create and use hyper-heuristics for the FFP
# This is a class you must extend it to provide the actual implementation
class HyperHeuristic:

  # Constructor
  #   features = A list with the names of the features to be used by this hyper-heuristic
  #   heuristics = A list with the names of the heuristics to be used by this hyper-heuristic
  #   model_selected = 0: Triangular Antecedents, 1: Trapezoidal Antecedents, 2: Gaussians Antecedents
  def __init__(self, features, heuristics,model_selected):
    if (features):
      self.features = features.copy()
    
    else:
      print("=====================")
      print("Critical error at HyperHeuristic.__init__.")
      print("The list of features cannot be empty.")
      print("The system will halt.")
      print("=====================")
      exit(0)
    if (heuristics):
      self.heuristics = heuristics.copy()
      #super().__init__(features, heuristics)

      # Load the Fuzzy Model
      ## Define the Consequent (The Heuristics)
      self.fuzzyHH = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'fuzzyHH')
      self.fuzzyHH['LDEG'] = fuzz.gaussmf(self.fuzzyHH.universe, 0.3,0.2)
      self.fuzzyHH['GDEG'] = fuzz.gaussmf(self.fuzzyHH.universe, 0.7,0.2)
      ### You can change the defuzzification method 'centroid','bisector','mom','som','lom'
      ### Where they mean: mean of maximum, min of maximum, max of maximum
      self.fuzzyHH.defuzzify_method = 'centroid'
      
      ## Define the Antecedents
      ## 4 Features included in the Fire figther problem
      ## Triangular Antecedent Model
      if model_selected == 0:
        ### 1. Edge Density (Represented with two member functions)
        self.ED = ctrl.Antecedent(np.arange(0, 1.0, 0.01), 'ED')
        self.ED['EDL'] = fuzz.trimf(self.ED.universe, [0.0, 0.0, 1.0])
        self.ED['EDH'] = fuzz.trimf(self.ED.universe, [0.0, 1.0, 1.0])

        ### 2. Burning Nodes (Represented with two member functions)
        self.BN = ctrl.Antecedent(np.arange(0, 1.0, 0.01), 'BN')
        self.BN['BNL'] = fuzz.trimf(self.BN.universe, [0.0, 0.0, 1.0])
        self.BN['BNH'] = fuzz.trimf(self.BN.universe, [0.0, 1.0, 1.0])

        ### 3. Nodes in Danger (Represented with two member functions)
        self.ND = ctrl.Antecedent(np.arange(0, 1.0, 0.01), 'ND')
        self.ND['NDL'] = fuzz.trimf(self.ND.universe, [0.0, 0.0, 1.0])
        self.ND['NDH'] = fuzz.trimf(self.ND.universe, [0.0, 1.0, 1.0])

        ### 4. Average Degree (Represented with two member functions)
        self.AD = ctrl.Antecedent(np.arange(0, 1.0, 0.01), 'AD')
        self.AD['ADL'] = fuzz.trimf(self.AD.universe, [0.0, 0.0, 1.0])
        self.AD['ADH'] = fuzz.trimf(self.AD.universe, [0.0, 1.0, 1.0])

      ## Trapezoidal Antecedent Model
      elif model_selected == 1:
        ### 1. Edge Density (Represented with two member functions)
        self.ED = ctrl.Antecedent(np.arange(0, 1.0, 0.01), 'ED')
        self.ED['EDL'] = fuzz.trapmf(self.ED.universe, [0, 0, 0.3, 1])
        self.ED['EDH'] = fuzz.trapmf(self.ED.universe, [0, 0.7, 1, 1])

        ### 2. Burning Nodes (Represented with two member functions)
        self.BN = ctrl.Antecedent(np.arange(0, 1.0, 0.01), 'BN')
        self.BN['BNL'] = fuzz.trapmf(self.BN.universe, [0, 0, 0.3, 1])
        self.BN['BNH'] = fuzz.trapmf(self.BN.universe, [0, 0.7, 1, 1])

        ### 3. Nodes in Danger (Represented with two member functions)
        self.ND = ctrl.Antecedent(np.arange(0, 1.0, 0.01), 'ND')
        self.ND['NDL'] = fuzz.trapmf(self.ND.universe, [0, 0, 0.3, 1])
        self.ND['NDH'] = fuzz.trapmf(self.ND.universe, [0, 0.7, 1, 1])

        ### 4. Average Degree (Represented with two member functions)
        self.AD = ctrl.Antecedent(np.arange(0, 1.0, 0.01), 'AD')
        self.AD['ADL'] = fuzz.trapmf(self.AD.universe, [0, 0, 0.3, 1])
        self.AD['ADH'] = fuzz.trapmf(self.AD.universe, [0, 0.7, 1, 1])
      
      ## Gaussian Antecedent Model
      elif model_selected == 2:
        ### 1. Edge Density (Represented with two member functions)
        self.ED = ctrl.Antecedent(np.arange(0, 1.0, 0.01), 'ED')
        self.ED['EDL'] = fuzz.gaussmf(self.ED.universe, 0.0,0.3) # mean, std
        self.ED['EDH'] = fuzz.gaussmf(self.ED.universe, 1.0,0.3)

        ### 2. Burning Nodes (Represented with two member functions)
        self.BN = ctrl.Antecedent(np.arange(0, 1.0, 0.01), 'BN')
        self.BN['BNL'] = fuzz.gaussmf(self.BN.universe, 0.0,0.3)
        self.BN['BNH'] = fuzz.gaussmf(self.BN.universe, 1.0,0.3)

        ### 3. Nodes in Danger (Represented with two member functions)
        self.ND = ctrl.Antecedent(np.arange(0, 1.0, 0.01), 'ND')
        self.ND['NDL'] = fuzz.gaussmf(self.ND.universe, 0.0,0.3)
        self.ND['NDH'] = fuzz.gaussmf(self.ND.universe, 1.0,0.3)

        ### 4. Average Degree (Represented with two member functions)
        self.AD = ctrl.Antecedent(np.arange(0, 1.0, 0.01), 'AD')
        self.AD['ADL'] = fuzz.gaussmf(self.AD.universe, 0.0,0.3)
        self.AD['ADH'] = fuzz.gaussmf(self.AD.universe, 1.0,0.3)

      ## Define the Fuzzy Rules
      ### 16 Fuzzy Rules defined for the 4 features (One for the LDGE and another one for the GDGE)
      rule1  = ctrl.Rule(self.ED['EDL'] & self.BN['BNL'] & self.ND['NDL'] & self.AD['ADL'], self.fuzzyHH['LDEG'])
      rule2  = ctrl.Rule(self.ED['EDL'] & self.BN['BNL'] & self.ND['NDL'] & self.AD['ADH'], self.fuzzyHH['GDEG'])
      rule3  = ctrl.Rule(self.ED['EDL'] & self.BN['BNL'] & self.ND['NDH'] & self.AD['ADL'], self.fuzzyHH['LDEG'])
      rule4  = ctrl.Rule(self.ED['EDL'] & self.BN['BNL'] & self.ND['NDH'] & self.AD['ADH'], self.fuzzyHH['GDEG'])
      rule5  = ctrl.Rule(self.ED['EDL'] & self.BN['BNH'] & self.ND['NDL'] & self.AD['ADL'], self.fuzzyHH['LDEG'])
      rule6  = ctrl.Rule(self.ED['EDL'] & self.BN['BNH'] & self.ND['NDL'] & self.AD['ADH'], self.fuzzyHH['GDEG'])
      rule7  = ctrl.Rule(self.ED['EDL'] & self.BN['BNH'] & self.ND['NDH'] & self.AD['ADL'], self.fuzzyHH['LDEG'])
      rule8  = ctrl.Rule(self.ED['EDL'] & self.BN['BNH'] & self.ND['NDH'] & self.AD['ADH'], self.fuzzyHH['GDEG'])
      rule9  = ctrl.Rule(self.ED['EDH'] & self.BN['BNL'] & self.ND['NDL'] & self.AD['ADL'], self.fuzzyHH['LDEG'])
      rule10 = ctrl.Rule(self.ED['EDH'] & self.BN['BNL'] & self.ND['NDL'] & self.AD['ADH'], self.fuzzyHH['GDEG'])
      rule11 = ctrl.Rule(self.ED['EDH'] & self.BN['BNL'] & self.ND['NDH'] & self.AD['ADL'], self.fuzzyHH['LDEG'])
      rule12 = ctrl.Rule(self.ED['EDH'] & self.BN['BNL'] & self.ND['NDH'] & self.AD['ADH'], self.fuzzyHH['GDEG'])
      rule13 = ctrl.Rule(self.ED['EDH'] & self.BN['BNH'] & self.ND['NDL'] & self.AD['ADL'], self.fuzzyHH['LDEG'])
      rule14 = ctrl.Rule(self.ED['EDH'] & self.BN['BNH'] & self.ND['NDL'] & self.AD['ADH'], self.fuzzyHH['GDEG'])
      rule15 = ctrl.Rule(self.ED['EDH'] & self.BN['BNH'] & self.ND['NDH'] & self.AD['ADL'], self.fuzzyHH['LDEG'])
      rule16 = ctrl.Rule(self.ED['EDH'] & self.BN['BNH'] & self.ND['NDH'] & self.AD['ADH'], self.fuzzyHH['GDEG'])
      
      ### Combine all the rules
      self.rules = ctrl.ControlSystem([ rule1, rule2, rule3, rule4, 
                                        rule5, rule6, rule7, rule8, 
                                        rule9, rule10, rule11, rule12,
                                        rule13,rule14, rule15, rule16])

      ### In order to simulate this control system, we will create a ``ControlSystemSimulation`` which represent the fuzzy logic system
      self.ffp_inference = ctrl.ControlSystemSimulation(self.rules)

    else:
      print("=====================")
      print("Critical error at HyperHeuristic.__init__.")
      print("The list of heuristics cannot be empty.")
      print("The system will halt.")
      print("=====================")
      exit(0)
  
  # Returns the next heuristic to use
  #   problem = The FFP instance being solved
  def nextHeuristic(self, problem):
    index = -1
    # Represents the features of the current state
    state = []
    for i in range(len(self.features)):
      state.append(problem.getFeature(self.features[i]))
    #print("\t" + str(state))

    # Provide feature inputs to the Fuzzy Hyper Heuristic
    ## [0]: Edge_Density [1]: Burning Nodes [2]: Nodes in Danger [3]: Avg Degree 
    self.ffp_inference.input['ED'] = state[0]
    self.ffp_inference.input['BN'] = state[1]
    self.ffp_inference.input['ND'] = state[2]
    self.ffp_inference.input['AD'] = state[3]

    ## Compute the output with the given values
    self.ffp_inference.compute()

    ## Result of the Fuzzy Logic System (value from 0 to 1)
    heuristic_value = self.ffp_inference.output['fuzzyHH']
    #print("Output of the FHH:", heuristic_value)
    ## Decide which heuristic to use based on threshold
    if heuristic_value < 0.5:
      heuristic = "LDEG"
      index=0
    else: 
      heuristic = "GDEG" 
      index=1
    #print("\t\t=> " + str(heuristic) + " (R" + str(index) + ")")
    return heuristic


  # Returns the string representation of this hyper-heuristic 
  def __str__(self):
    text = "Features:\n\t" + str(self.features) + "\nHeuristics:\n\t" + str(self.heuristics) + "\nRules:\n" + str(self.rules.rules.all_rules)
    return text

# A dummy hyper-heuristic for testing purposes.
# The hyper-heuristic creates a set of randomly initialized rules.
# Then, when called, it measures the distance between the current state and the
# conditions in the rules
# The rule with the condition closest to the problem state is the one that fires
class DummyHyperHeuristic(HyperHeuristic):

  # Constructor
  #   features = A list with the names of the features to be used by this hyper-heuristic
  #   heuristics = A list with the names of the heuristics to be used by this hyper-heuristic
  #   nbRules = The number of rules to be contained in this hyper-heuristic  
  def __init__(self, features, heuristics, nbRules, seed):
    super().__init__(features, heuristics)
    random.seed(seed)
    self.conditions = []
    self.actions = []
    for i in range(nbRules):
      self.conditions.append([0] * len(features))
      for j in range(len(features)):
        self.conditions[i][j] = random.random()
      self.actions.append(heuristics[random.randint(0, len(heuristics) - 1)])
  
  # Returns the next heuristic to use
  #   problem = The FFP instance being solved
  def nextHeuristic(self, problem):
    minDistance = float("inf")
    index = -1
    state = []
    for i in range(len(self.features)):
      state.append(problem.getFeature(self.features[i]))
    print("\t" + str(state))
    for i in range(len(self.conditions)):
      distance = self.__distance(self.conditions[i], state)      
      if (distance < minDistance):
        minDistance = distance
        index = i
    heuristic = self.actions[index] 
    print("\t\t=> " + str(heuristic) + " (R" + str(index) + ")")
    return heuristic

  # Returns the string representation of this dummy hyper-heuristic
  def __str__(self):
    text = "Features:\n\t" + str(self.features) + "\nHeuristics:\n\t" + str(self.heuristics) + "\nRules:\n"
    for i in range(len(self.conditions)):      
      text += "\t" + str(self.conditions[i]) + " => " + self.actions[i] + "\n"      
    return text

  # Returns the Euclidian distance between two vectors
  def __distance(self, vectorA, vectorB):
    distance = 0
    for i in range(len(vectorA)):
      distance += (vectorA[i] - vectorB[i]) ** 2
    distance = math.sqrt(distance)
    return distance

# Tests
# =====================
# Store the Heuristics and FHH Burning Nodes
df = pd.DataFrame(columns=['LDEG','GDEG','Fuzzy_Hyper_Heuristic'])

# Paths of the instances
trainset_path = './instances/BBGRL/'
testset_path = './instances/GBRL/'

# Retrieve the files from the path
instances = os.listdir(trainset_path)
# Test the heuristics and FHH for all the instances
for instance in instances:
  # Solves the problem using heuristic LDEG and one firefighter
  problem = FFP(trainset_path+instance) # Path to a specific File
  #print("LDEG = " + str(problem.solve("LDEG", 1, False)))
  result_1 = problem.solve("LDEG", 1, False)
  
  # Solves the problem using heuristic GDEG and one firefighter
  problem = FFP(trainset_path+instance) # Path to a specific File
  #print("GDEG = " + str(problem.solve("GDEG", 1, False)))
  result_2 = problem.solve("GDEG", 1, False)

  # Solves the problem using Fuzzy Hyper Heuristic and one firefighter
  problem = FFP(trainset_path+instance)
  ffp_hh_obj = HyperHeuristic(["EDGE_DENSITY", "BURNING_NODES", "NODES_IN_DANGER", "AVG_DEGREE"], ["LDEG", "GDEG"], model_selected=0)
  #print("Fuzzy HH = " + str(problem.solve(ffp_hh_obj, 1, False)))
  result_3 = problem.solve(ffp_hh_obj, 1, False)
  
  #store the information in the dataframe 
  df = df.append({'LDEG': result_1, 'GDEG': result_2, 'Fuzzy_Hyper_Heuristic': result_3}, ignore_index=True)

df.to_csv('trainset_results.csv')