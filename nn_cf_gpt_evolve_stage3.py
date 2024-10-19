import torch
import numpy as np
import sys
import random
import os
import re
import logging
import matplotlib.pyplot as plt
import pickle
import time
from torch import nn
from torch.optim import Adam, AdamW
import torch.nn.functional as F
import copy
import glob
import gc
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
import math
from nn_cf_gpt import GPTLanguageModel, get_batch



# gpt hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 10000
eval_interval = 10
learning_rate = 3e-4
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
#print('device:',device)
eval_iters = 16
train_iters = 16
n_embd = 32
n_head = 3
n_layer = 3
dropout = 0.15


#evolution metaparameters
pop_size = 70
k_top = 20
k_safe = 5
mr1 = 0.03
mr2 = 0.001
load_loss_weigths = True

#tests:
#pop_size=30,70,100,150,300,1000
#k_top = 5,10,20,30,50,70
#k_safe = 1,3,5,10,20
#mr1 = 0.3 0.1 0.03 0.01
#mr2 = 0.1 0.03 0.003 0.001
#seed
seed = 740
hisotry_file_path = '/Users/danilkutny/Desktop/ai_work/backpop_research/nn_cost_funct/data/evo_results/history'
# ------------

torch.manual_seed(seed)
np.random.seed(seed)


@torch.no_grad()
def evaluate_model(model, eval_data):
    model.eval()
    loss_total = 0
    for indx, labels in eval_data:
        logits, loss_pred, loss_real = model(indx, labels)
        loss_total+=loss_real
    loss_total = loss_total/len(eval_data)
    return loss_total.item()
    

def initialize_population(pop_size, magic_scale=0.1, pop_file=None):#magic_scale=0.112
    population = []
    
    if pop_file == None:
        for _ in range(pop_size):
            model = GPTLanguageModel(load_loss_weigths=load_loss_weigths, add_loss_noise=True)
            for p in model.loss.parameters():
                p.data = p.data + torch.randn(p.data.shape)*random.randint(1,200)/100000# 0.00001#0.003-0.01 best!, use 0.001 to 0.02
            individual = model.loss.state_dict().copy()

            population.append(individual)
        return population
    
    else:
        with open(pop_file, 'rb') as file:
            pop_cpu = pickle.load(file)
        pop = []
        for ind in pop_cpu:
            pop.append({})
            for name, params in ind.items():
                pop[-1][name] = [par.to(device) for par in params]
        if len(pop)<pop_size:
            while len(pop)<pop_size:
                pop = pop+pop
        pop = pop[:pop_size]
        del pop_cpu
        for name, model_inst in lrs.items():
            model_inst.to(device)
        return pop


def mutate(weights, mr1=mr1, mr2=mr2, seed=None):
    while True:
        random_w_i = random.randint(0, len(weights) - 1)
        if random.randint(0, 3) == 0:
            mr2 = mr2 * 3
            if random.randint(0, 3) == 0:
                mr2 = mr2 * 3
                if random.randint(0, 3) == 0:
                    mr2 = mr2 * 10
        
        mutation_mask = torch.rand(weights[random_w_i].shape) < max(mr1, 1.0 / weights[random_w_i].numel())
        weights[random_w_i] += mutation_mask * torch.randn(weights[random_w_i].shape) * mr2
        if random.randint(0, 1) == 0:
            break
    return weights


def select_top_population(population, fitness_scores, top_k=k_top, minimize=True):
    if minimize:
        sorted_indices = np.argsort(fitness_scores)[:top_k]
        best = fitness_scores[sorted_indices[0]]
    else:
        sorted_indices = np.argsort(fitness_scores)[-top_k:]
        best = fitness_scores[sorted_indices[-1]]
    
    #print('sorted, chosen first:',sorted_indices)
    new_pop = [population[i] for i in sorted_indices]
    return new_pop, best


def crossover(parent1, parent2, seed=None):
    child = []
    for p1, p2 in zip(parent1, parent2):
        mask = np.random.rand(*p1.shape) > 0.5
        child.append(torch.where(torch.tensor(mask, dtype=torch.bool), p1, p2))
    return child


def generate_new_population(selected_population, pop_size=pop_size, k_safe=k_safe, mutation_rate=0.03, seed=None, minimize=True):
    new_population = []
    if minimize:
        safe_k_pop = selected_population[:k_safe]
    else:
        safe_k_pop = selected_population[-k_safe:]
    for individual in safe_k_pop:
        new_population.append({})
        for spec_name in safe_k_pop[0].keys():
            new_population[-1][spec_name] = individual[spec_name].clone().detach()

    selected_indices = np.arange(len(selected_population))
    while len(new_population) < pop_size:
        new_population.append({})
        parent_indices = np.random.choice(selected_indices, 2, replace=False)
        parent1 = selected_population[parent_indices[0]]
        parent2 = selected_population[parent_indices[1]]
        parent1, parent2 = [w.clone().detach().to('cpu') for w in parent1.values()], [w.clone().detach().to('cpu') for w in parent2.values()]
        child_weights = crossover(parent1, parent2, seed=seed)
        child_weights = mutate(child_weights, mutation_rate, seed=seed)
        for n, w in zip(safe_k_pop[0].keys(), child_weights):
            new_population[-1][n] = w.clone().detach().to(device)
    return new_population


class FitnessSimulation():
    def __init__(self, model):
        ideal = {}
        for (n, w) in model.loss.state_dict().items():
            ideal[n] = torch.randn(w.shape)*2
        self.ideal = ideal
        self.model = model
    
    def eval(self, pop):
        fitness = []
        for ind in pop:
            ind_val = 0
            for n in ind.keys():
                #print(self.ideal[n].device, ind[n].device)
                w_v = (self.ideal[n].to(ind[n].device)-ind[n])**2
                ind_val+=w_v.mean().item()
            fitness.append(ind_val)
        return fitness

    #def metaparameter_search(self, pop):
    #    self.pop = pop
    #    #for i in range(300):
    #    #fitness_scores = env.eval(pop)
    #    #selcted_pop, best = select_top_population(pop, fitness_scores)






model = GPTLanguageModel(load_loss_weigths=load_loss_weigths, add_loss_noise=True)
m = model.to(device)


#ideal = {}
#for n, w in model.loss.state_dict():
#    ideal[n] = torch.randn(w.shape)*2
# print the number of parameters in the model
#print(sum(p.numel() for p in m.parameters()), 'parameters')

#ideal_ind = 

#pop = initialize_population(pop_size)
gen_to_load = 'gen_1220.pkl'
saved_gens_file_path = f'/Users/danilkutny/Desktop/ai_work/backpop_research/nn_cost_funct/data/saved_gens/{gen_to_load}'
with open(saved_gens_file_path, 'rb') as file:
    pop = pickle.load(file)
#env = FitnessSimulation(model)
#try:
with open(hisotry_file_path, 'rb') as file:
    history = pickle.load(file)
#history = []
for gen in range(1221, 99999999):
    #if gen%10 == 0:
    #    train_iters = train_iters+1
    #    eval_iters = eval_iters+1
    current_history = {}
    #fitness_scores = env.eval(pop)
    fitness_scores = []
    model = GPTLanguageModel(load_loss_weigths=load_loss_weigths, add_loss_noise=False).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    #extarct radnom numbers random numbers:
    folder_path = '/Users/danilkutny/Desktop/ai_work/backpop_research/nn_cost_funct/data/trained_gpts'
    pattern = re.compile(r'(?:model)(\d+)')
    # Initialize a list to store the extracted numbers
    extracted_numbers = []
    # Loop through each file in the folder
    for filename in os.listdir(folder_path):
        match = pattern.match(filename)  # Try to match the pattern
        if match:
            # Extract the number (as a string) and convert it to an integer
            extracted_number = int(match.group(1))
            extracted_numbers.append(extracted_number)
    # Sort the numbers for convenience (optional)
    extracted_numbers.sort()
    rnd_numb = random.choice(extracted_numbers)

    start_weigths = torch.load(f'/Users/danilkutny/Desktop/ai_work/backpop_research/nn_cost_funct/data/trained_gpts/model{rnd_numb}')
    optimzer_state_dict = torch.load(f'/Users/danilkutny/Desktop/ai_work/backpop_research/nn_cost_funct/data/trained_gpts/optimizer{rnd_numb}')

    model.load_state_dict( copy.deepcopy(start_weigths) )
    optimizer.load_state_dict( copy.deepcopy(optimzer_state_dict) )
    #start_weigths = model.state_dict()
    
    #additional training for randomndes:
    for _ in range(8):
        x, y = get_batch('train')
        optimizer.zero_grad(set_to_none=True)
        logits, loss, loss2 = model(x, y)
        loss2.backward()
        optimizer.step()

    start_weigths = copy.deepcopy(model.state_dict())
    optimzer_state_dict = copy.deepcopy(optimizer.state_dict())


    train_data = [get_batch('train') for i in range(train_iters)]
    eval_data = [get_batch('val') for i in range(eval_iters)]
    
    #real cross_entropy
    #model.to(device)
    for x, y in train_data:
        optimizer.zero_grad(set_to_none=True)
        logits, loss, loss2 = model(x, y)
        loss2.backward()
        optimizer.step()
        #print(loss2.item())
    #sys.exit()
    control_loss = evaluate_model(model, eval_data)
    #print('control_loss:',control_loss)
    current_history['control'] = control_loss
    del model
    #evaluate
    #last_ind = pop[0]
    for ind in pop:

            #print(w1.shape,w2.shape)
            #sys.exit()
        #inizilize model, set current train iter weitghs and current evo loss weigths adn feeze los weight 
        model = GPTLanguageModel(load_loss_weigths=False, add_loss_noise=False).to(device)
        model.load_state_dict( copy.deepcopy(start_weigths) )
        model.set_loss_weights( ind )
        model.loss_weigths_freeze()
        all_true = True
        cpu_mps = ''
        '''
        for w1, w2 in zip(model.loss.state_dict().values(), last_ind.values()):
            try:
                all_true = torch.equal(w1, w2)
                if all_true==False:
                    #print(w1)
                    #print(w2)
                    break
            except RuntimeError:
                w1_device = str(w1.device)
                w2_device = str(w2.device)
                all_true = torch.equal(w1.to('cpu'), w2.to('cpu'))
                cpu_mps = 'CPU MPS DISMATCH'
                w1.to(w1_device), w2.to(w2_device)
                if all_true==False:
                    #print(w1)
                    #print(w2)
                    break
        '''

        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        optimizer.load_state_dict( copy.deepcopy(optimzer_state_dict) )
        for x, y in train_data:
            optimizer.zero_grad(set_to_none=True)
            logits, loss, loss2 = model(x, y)
            loss.backward()
            optimizer.step()
        fitness_score = evaluate_model(model, eval_data)
        fitness_scores.append(fitness_score)
        #print(fitness_score)
        #last_ind = dict([(n, w.detach().clone()) for n,w in model.loss.state_dict().items()])
        del model
    #print(fitness_scores)


    current_history['results'] = fitness_scores
    history.append(current_history)
    with open(hisotry_file_path, 'wb') as file:
        pickle.dump(history, file)
    if gen%10==0:
        #pass
        saved_gens_file_path = f'/Users/danilkutny/Desktop/ai_work/backpop_research/nn_cost_funct/data/saved_gens/gen_{gen}.pkl'
        with open(saved_gens_file_path, 'wb') as file:
            pickle.dump(pop, file)
    selcted_pop, best = select_top_population(pop, fitness_scores)
    print(f'#{gen} | steps: {train_iters} | control: {control_loss:.4f} | res: {best:.4f}')
    pop = generate_new_population(selcted_pop)
    gc.collect()
    #print('done')
    #print(pop[0].keys())
print('fitness_scores:',sum(fitness_scores)/len(fitness_scores))




sys.exit()






























