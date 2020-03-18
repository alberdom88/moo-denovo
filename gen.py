import torch
import numpy as np
import time
import os

from model import RNN
from data_structs import Vocabulary, Experience
from scoring_functions import get_scoring_function
from utils import Variable, seq_to_smiles, fraction_valid_smiles, unique
from vizard_logger import VizardLog

def train_agent(restore_prior_from='Prior.ckpt',
                restore_agent_from='Prior.ckpt',
                scoring_function='logp_mw',
                scoring_function_kwargs=None,
                save_dir=None, learning_rate=0.0005,
                batch_size=64, n_steps=3000,
                num_processes=0, sigma=60,
                experience_replay=0):

    voc = Vocabulary(init_from_file="Voc")

    start_time = time.time()

    Prior = RNN(voc)
    Agent = RNN(voc)


    # By default restore Agent to same model as Prior, but can restore from already trained Agent too.
    # Saved models are partially on the GPU, but if we dont have cuda enabled we can remap these
    # to the CPU.
    #if torch.cuda.is_available():
    if False:
        Prior.rnn.load_state_dict(torch.load('Prior.ckpt'))
        Agent.rnn.load_state_dict(torch.load(restore_agent_from))
    else:
        prior_tmp=torch.load('Prior.ckpt', map_location=lambda storage, loc: storage)
        prior_tmp['gru_1.bias_ih']=prior_tmp['gru_1.bias_ih'].reshape(1536)
        prior_tmp['gru_1.bias_hh']=prior_tmp['gru_1.bias_hh'].reshape(1536)
        prior_tmp['gru_2.bias_ih']=prior_tmp['gru_2.bias_ih'].reshape(1536)
        prior_tmp['gru_2.bias_hh']=prior_tmp['gru_2.bias_hh'].reshape(1536)
        prior_tmp['gru_3.bias_ih']=prior_tmp['gru_3.bias_ih'].reshape(1536)
        prior_tmp['gru_3.bias_hh']=prior_tmp['gru_3.bias_hh'].reshape(1536)
        Prior.rnn.load_state_dict(prior_tmp)
        agent_tmp=torch.load(restore_agent_from, map_location=lambda storage, loc: storage)
        agent_tmp['gru_1.bias_ih']=agent_tmp['gru_1.bias_ih'].reshape(1536)
        agent_tmp['gru_1.bias_hh']=agent_tmp['gru_1.bias_hh'].reshape(1536)
        agent_tmp['gru_2.bias_ih']=agent_tmp['gru_2.bias_ih'].reshape(1536)
        agent_tmp['gru_2.bias_hh']=agent_tmp['gru_2.bias_hh'].reshape(1536)
        agent_tmp['gru_3.bias_ih']=agent_tmp['gru_3.bias_ih'].reshape(1536)
        agent_tmp['gru_3.bias_hh']=agent_tmp['gru_3.bias_hh'].reshape(1536)
        Agent.rnn.load_state_dict(agent_tmp)

    # We dont need gradients with respect to Prior
    for param in Prior.rnn.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(Agent.rnn.parameters(), lr=0.0005)

    # Scoring_function
    scoring_function = get_scoring_function(scoring_function=scoring_function, num_processes=num_processes,
                                            **scoring_function_kwargs)

    # For policy based RL, we normally train on-policy and correct for the fact that more likely actions
    # occur more often (which means the agent can get biased towards them). Using experience replay is
    # therefor not as theoretically sound as it is for value based RL, but it seems to work well.
    experience = Experience(voc)

    # Information for the logger
    step_score = [[], []]

    print("Model initialized, starting training...")
    
    if not save_dir:
       save_dir = 'run_' + time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
    os.makedirs(save_dir, exist_ok=True)
    logger = VizardLog(save_dir)
    
    # Log some network weights that can be dynamically plotted with the Vizard bokeh app
    logger.log(Agent.rnn.gru_2.weight_ih.cpu().data.numpy()[::100], "init_weight_GRU_layer_2_w_ih")
    logger.log(Agent.rnn.gru_2.weight_hh.cpu().data.numpy()[::100], "init_weight_GRU_layer_2_w_hh")
    logger.log(Agent.rnn.embedding.weight.cpu().data.numpy()[::30], "init_weight_GRU_embedding")
    logger.log(Agent.rnn.gru_2.bias_ih.cpu().data.numpy(), "init_weight_GRU_layer_2_b_ih")
    logger.log(Agent.rnn.gru_2.bias_hh.cpu().data.numpy(), "init_weight_GRU_layer_2_b_hh")    

    score_old = 0.0
    for step in range(n_steps):

        # Sample from Agent
        seqs, agent_likelihood, entropy = Agent.sample(batch_size)

        # Remove duplicates, ie only consider unique seqs
        unique_idxs = unique(seqs)
        seqs = seqs[unique_idxs]
        agent_likelihood = agent_likelihood[unique_idxs]
        entropy = entropy[unique_idxs]

        # Get prior likelihood and score
        prior_likelihood, _ = Prior.likelihood(Variable(seqs))
        smiles = seq_to_smiles(seqs, voc)
        score = scoring_function(smiles)
        
        #convert des_rels
        rels = []
        for ccc in scoring_function_kwargs['des_rels']:
                if (ccc=='Maximize'):
                   rels.append(False)
                if (ccc=='Minimize'):
                   rels.append(True)
                   
 
        if (len(score[0])==2):
            with open(os.path.join(save_dir, "gen_%s.txt" %(step)),"w") as fff:
                 for iii in range(len(smiles)):
                     fff.write("%s %s %s\n" %(smiles[iii],score[iii][0],score[iii][1]))
            score = torch.Tensor(np.array(score)[:,1])         
        else:  
             no_v = []
             for aaa in range(len(score)):
                 if (score[aaa][-1]<(len(score[aaa])-1.0)):
                     no_v.append(aaa) 
             domv = []
             domat = []
             for aaa in range(len(score)):
                 dom = np.zeros(int((len(score[0])-1)*(len(score[0])-2)/2))
                 domat.append(dom)
             for aaa in range(len(score)):
                 if aaa in no_v:
                    domv.append(score[aaa][-1])
                    continue
                 for bbb in range(aaa+1,len(score)):
                     if (aaa==bbb):
                        continue
                     if bbb in no_v:
                        continue
                     ab = np.array(score[aaa])>=np.array(score[bbb])
                     ba = np.array(score[bbb])>=np.array(score[aaa])
                     uu = np.array(score[aaa])==np.array(score[bbb])
                     ccc = 0
                     for iii in range(len(score[0])-1):
                         for jjj in range(iii+1,len(score[0])-1):
                              ab_check = []
                              ba_check = []
                              v_rels = []
                              if (uu[iii]==False):
                                 ab_check.append(ab[iii])
                                 ba_check.append(ba[iii])
                                 v_rels.append(rels[iii])
                              if (uu[jjj]==False):
                                 ab_check.append(ab[jjj])
                                 ba_check.append(ba[jjj])
                                 v_rels.append(rels[jjj])
                              if (len(v_rels)==0):
                                 continue            
                              if (np.all(ab_check==v_rels)):
                                  domat[aaa][ccc]+=1
                              if (np.all(ba_check==v_rels)):
                                 domat[bbb][ccc]+=1
                              ccc = ccc + 1
                 dom_final = 0.0
                 for iii in range(len(domat[aaa])):
                     dom_final=dom_final+float(batch_size-len(no_v)-domat[aaa][iii])/float(batch_size-len(no_v))
                 domv.append(dom_final/len(domat[aaa])+score[aaa][-1])
             with open(os.path.join(save_dir, "gen_%s.txt" %(step)),"w") as fff:
                  for iii in range(len(smiles)):
                      sc = ""
                      for kkk in range(len(score[iii])):
                          sc = sc + str(score[iii][kkk]) + " "
                      fff.write("%s %s %s\n" %(smiles[iii],sc,domv[iii]))
             score = torch.Tensor(np.array(domv))
        
            

        ######################

        # Calculate augmented likelihood
        augmented_likelihood = prior_likelihood + sigma * Variable(score)
        loss = torch.pow((augmented_likelihood - agent_likelihood), 2)

        # Experience Replay
        # First sample
        if experience_replay and len(experience)>4:
            exp_seqs, exp_score, exp_prior_likelihood = experience.sample(4)
            exp_agent_likelihood, exp_entropy = Agent.likelihood(exp_seqs.long())
            exp_augmented_likelihood = exp_prior_likelihood + sigma * exp_score
            exp_loss = torch.pow((Variable(exp_augmented_likelihood) - exp_agent_likelihood), 2)
            loss = torch.cat((loss, exp_loss), 0)
            agent_likelihood = torch.cat((agent_likelihood, exp_agent_likelihood), 0)

        # Then add new experience
        prior_likelihood = prior_likelihood.data.cpu().numpy()
        new_experience = zip(smiles, score, prior_likelihood)
        experience.add_experience(new_experience)

        # Calculate loss
        loss = loss.mean()

        # Add regularizer that penalizes high likelihood for the entire sequence
        loss_p = - (1 / agent_likelihood).mean()
        loss += 5 * 1e3 * loss_p

        # Calculate gradients and make an update to the network weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Convert to numpy arrays so that we can print them
        augmented_likelihood = augmented_likelihood.data.cpu().numpy()
        agent_likelihood = agent_likelihood.data.cpu().numpy()

        # Print some information for this step
        time_elapsed = (time.time() - start_time) / 3600
        time_left = (time_elapsed * ((n_steps - step) / (step + 1)))
        print("\n       Step {}   Fraction valid SMILES: {:4.1f}  Time elapsed: {:.2f}h Time left: {:.2f}h".format(
              step, fraction_valid_smiles(smiles) * 100, time_elapsed, time_left))
        print("  Agent    Prior   Target   Score             SMILES")
        for i in range(10):
            print(" {:6.2f}   {:6.2f}  {:6.2f}  {:6.2f}     {}".format(agent_likelihood[i],
                                                                       prior_likelihood[i],
                                                                       augmented_likelihood[i],
                                                                       score[i],
                                                                       smiles[i]))
        # Need this for Vizard plotting
        step_score[0].append(step + 1)
        step_score[1].append(np.mean(np.array(score)))

        # Log some weights
        logger.log(Agent.rnn.gru_2.weight_ih.cpu().data.numpy()[::100], "weight_GRU_layer_2_w_ih")
        logger.log(Agent.rnn.gru_2.weight_hh.cpu().data.numpy()[::100], "weight_GRU_layer_2_w_hh")
        logger.log(Agent.rnn.embedding.weight.cpu().data.numpy()[::30], "weight_GRU_embedding")
        logger.log(Agent.rnn.gru_2.bias_ih.cpu().data.numpy(), "weight_GRU_layer_2_b_ih")
        logger.log(Agent.rnn.gru_2.bias_hh.cpu().data.numpy(), "weight_GRU_layer_2_b_hh")
        logger.log("\n".join([smiles + "\t" + str(round(float(score), 2)) for smiles, score in zip \
                            (smiles[:12], score[:12])]), "SMILES", dtype="text", overwrite=True)
        logger.log(np.array(step_score), "Scores")
        score_new = np.mean(np.array(score))
        if (score_new > score_old):
           torch.save(Agent.rnn.state_dict(), os.path.join(save_dir, 'Agent_best.ckpt'))
           score_old = score_new
        if (step%50==0):
           experience.print_memory(os.path.join(save_dir, "memory"))
           torch.save(Agent.rnn.state_dict(), os.path.join(save_dir, 'Agent_%s.ckpt' %(step)))

    # If the entire training finishes, we create a new folder where we save this python file
    # as well as some sampled sequences and the contents of the experinence (which are the highest
    # scored sequences seen during training)
    
    #if not save_dir:
    #    save_dir = 'data/results/run_' + time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
    #os.makedirs(save_dir)

    experience.print_memory(os.path.join(save_dir, "memory"))
    torch.save(Agent.rnn.state_dict(), os.path.join(save_dir, 'Agent.ckpt'))

    seqs, agent_likelihood, entropy = Agent.sample(256)
    prior_likelihood, _ = Prior.likelihood(Variable(seqs))
    prior_likelihood = prior_likelihood.data.cpu().numpy()
    smiles = seq_to_smiles(seqs, voc)
    score = scoring_function(smiles)
    if (len(score[0])==2):
        with open(os.path.join(save_dir, "gen_final.txt"),"w") as fff:
             for iii in range(len(smiles)):
                 fff.write("%s %s %s\n" %(smiles[iii],score[iii][0],score[iii][1]))
        score = torch.Tensor(np.array(score)[:,1])
    else:
             no_v = []
             for aaa in range(len(score)):
                 if (score[aaa][-1]<(len(score[aaa])-1.0)):
                     no_v.append(aaa) 
             domv = []
             domat = []
             for aaa in range(len(score)):
                 dom = np.zeros(int((len(score[0])-1)*(len(score[0])-2)/2))
                 domat.append(dom)
             for aaa in range(len(score)):
                 if aaa in no_v:
                    domv.append(score[aaa][-1])
                    continue
                 for bbb in range(aaa+1,len(score)):
                     if (aaa==bbb):
                        continue
                     if bbb in no_v:
                        continue
                     ab = np.array(score[aaa])>=np.array(score[bbb])
                     ba = np.array(score[bbb])>=np.array(score[aaa])
                     uu = np.array(score[aaa])==np.array(score[bbb])
                     ccc = 0
                     for iii in range(len(score[0])-1):
                         for jjj in range(iii+1,len(score[0])-1):
                              ab_check = []
                              ba_check = []
                              v_rels = []
                              if (uu[iii]==False):
                                 ab_check.append(ab[iii])
                                 ba_check.append(ba[iii])
                                 v_rels.append(rels[iii])
                              if (uu[jjj]==False):
                                 ab_check.append(ab[jjj])
                                 ba_check.append(ba[jjj])
                                 v_rels.append(rels[jjj])
                              if (len(v_rels)==0):
                                 continue            
                              if (np.all(ab_check==v_rels)):
                                  domat[aaa][ccc]+=1
                              if (np.all(ba_check==v_rels)):
                                 domat[bbb][ccc]+=1
                              ccc = ccc + 1
                 dom_final = 0.0
                 for iii in range(len(domat[aaa])):
                     dom_final=dom_final+float(batch_size-len(no_v)-domat[aaa][iii])/float(batch_size-len(no_v))
                 domv.append(dom_final/len(domat[aaa])+score[aaa][-1])
             with open(os.path.join(save_dir, "gen_final.txt"),"w") as fff:
                 for iii in range(len(smiles)):
                     sc = ""
                     for kkk in range(len(score[iii])):
                         sc = sc + str(score[iii][kkk]) + " "
                     fff.write("%s %s %s\n" %(smiles[iii],sc,domv[iii]))
             score = torch.Tensor(np.array(domv))         

def sample_agent(restore_agent_from='Prior.ckpt',
                scoring_function='logp_mw',
                scoring_function_kwargs=None,
                save_dir=None,
                batch_size=64,
                num_processes=0):

    voc = Vocabulary(init_from_file="Voc")

    Agent = RNN(voc)

    # By default restore Agent to same model as Prior, but can restore from already trained Agent too.
    # Saved models are partially on the GPU, but if we dont have cuda enabled we can remap these
    # to the CPU.
    #if torch.cuda.is_available():
    if False:
        Agent.rnn.load_state_dict(torch.load(restore_agent_from))
    else:
        agent_tmp=torch.load(restore_agent_from, map_location=lambda storage, loc: storage)
        agent_tmp['gru_1.bias_ih']=agent_tmp['gru_1.bias_ih'].reshape(1536)
        agent_tmp['gru_1.bias_hh']=agent_tmp['gru_1.bias_hh'].reshape(1536)
        agent_tmp['gru_2.bias_ih']=agent_tmp['gru_2.bias_ih'].reshape(1536)
        agent_tmp['gru_2.bias_hh']=agent_tmp['gru_2.bias_hh'].reshape(1536)
        agent_tmp['gru_3.bias_ih']=agent_tmp['gru_3.bias_ih'].reshape(1536)
        agent_tmp['gru_3.bias_hh']=agent_tmp['gru_3.bias_hh'].reshape(1536)
        Agent.rnn.load_state_dict(agent_tmp)
    
    # We dont need gradients with respect to Prior
    scoring_function = get_scoring_function(scoring_function=scoring_function, num_processes=num_processes,
                                            **scoring_function_kwargs)

    # Information for the logger
    print("Model initialized, starting generating...")
    
    if not save_dir:
       save_dir = 'run_' + time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
    os.makedirs(save_dir, exist_ok=True)

    # Sample from Agent
    smiles_vec = []
    for ss in range(batch_size):
        print ('generating mol ',ss+1, 'out of ', batch_size)
        seqs, agent_likelihood, entropy = Agent.sample(1)
        smiles = seq_to_smiles(seqs, voc)
        smiles_vec.append(smiles[0])
    score = scoring_function(smiles_vec)
    #convert des_rels
    rels = []
    for ccc in scoring_function_kwargs['des_rels']:
            if (ccc=='Maximize'):
               rels.append(False)
            if (ccc=='Minimize'):
               rels.append(True)
               
    if (len(score[0])==2):
        with open(os.path.join(save_dir, "sampled.txt"),"w") as fff:
             for iii in range(len(smiles_vec)):
                 fff.write("%s %s %s\n" %(smiles_vec[iii],score[iii][0],score[iii][1]))
    else:  
             no_v = []
             for aaa in range(len(score)):
                 if (score[aaa][-1]<(len(score[aaa])-1.0)):
                     no_v.append(aaa) 
             domv = []
             domat = []
             for aaa in range(len(score)):
                 dom = np.zeros(int((len(score[0])-1)*(len(score[0])-2)/2))
                 domat.append(dom)
             for aaa in range(len(score)):
                 if aaa in no_v:
                    domv.append(score[aaa][-1])
                    continue
                 for bbb in range(aaa+1,len(score)):
                     if (aaa==bbb):
                        continue
                     if bbb in no_v:
                        continue
                     ab = np.array(score[aaa])>=np.array(score[bbb])
                     ba = np.array(score[bbb])>=np.array(score[aaa])
                     uu = np.array(score[aaa])==np.array(score[bbb])
                     ccc = 0
                     for iii in range(len(score[0])-1):
                         for jjj in range(iii+1,len(score[0])-1):
                              ab_check = []
                              ba_check = []
                              v_rels = []
                              if (uu[iii]==False):
                                 ab_check.append(ab[iii])
                                 ba_check.append(ba[iii])
                                 v_rels.append(rels[iii])
                              if (uu[jjj]==False):
                                 ab_check.append(ab[jjj])
                                 ba_check.append(ba[jjj])
                                 v_rels.append(rels[jjj])
                              if (len(v_rels)==0):
                                 continue            
                              if (np.all(ab_check==v_rels)):
                                  domat[aaa][ccc]+=1
                              if (np.all(ba_check==v_rels)):
                                 domat[bbb][ccc]+=1
                              ccc = ccc + 1
                 dom_final = 0.0
                 for iii in range(len(domat[aaa])):
                     dom_final=dom_final+float(batch_size-len(no_v)-domat[aaa][iii])/float(batch_size-len(no_v))
                 domv.append(dom_final/len(domat[aaa])+score[aaa][-1])
             with open(os.path.join(save_dir, "sampled.txt"),"w") as fff:
                  for iii in range(len(smiles_vec)):
                      sc = ""
                      for kkk in range(len(score[iii])-1):
                          sc = sc + str(score[iii][kkk]) + " "
                      fff.write("%s %s %s\n" %(smiles_vec[iii],sc,domv[iii]))
 


