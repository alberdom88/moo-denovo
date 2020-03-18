from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.ticker as ticker

from gen import train_agent, sample_agent
import multiprocessing as mp


import numpy as np
import os

des_list = ['MolWt','LogP','MolMR', 'SA', 'QED', 'Similarity', 'NumHDonors','NumHAcceptors','NumHeteroatoms', 
            'HeavyAtomCount', 'User_Fragment', 'NHOHCount','NOCount', 'CalcNumAtomStereoCenters', 'CalcNumAmideBonds', 'NumAliphaticCarbocycles', 
            'NumAliphaticHeterocycles', 'NumAliphaticRings', 'NumAromaticCarbocycles',
            'NumAromaticHeterocycles', 'NumAromaticRings', 'NumRotatableBonds',
            'NumSaturatedCarbocycles', 'NumSaturatedHeterocycles', 'NumSaturatedRings',
            'RingCount', 'FractionCSP3', 'fr_Al_COO', 'fr_Al_OH', 'fr_Al_OH_noTert','fr_ArN',
            'fr_Ar_COO','fr_Ar_N','fr_Ar_NH','fr_Ar_OH','fr_COO','fr_COO2','fr_C_O',
            'fr_C_O_noCOO','fr_C_S','fr_HOCCN','fr_Imine','fr_NH0','fr_NH1','fr_NH2','fr_N_O','fr_Ndealkylation1','fr_Ndealkylation2',
            'fr_Nhpyrrole','fr_SH','fr_aldehyde','fr_alkyl_carbamate','fr_alkyl_halide','fr_allylic_oxid',
            'fr_amide','fr_amidine','fr_aniline','fr_aryl_methyl','fr_azide','fr_azo',
            'fr_barbitur','fr_benzene','fr_benzodiazepine','fr_bicyclic','fr_diazo','fr_dihydropyridine',
            'fr_epoxide','fr_ester','fr_ether','fr_furan','fr_guanido','fr_halogen',
            'fr_hdrzine','fr_hdrzone','fr_imidazole','fr_imide','fr_isocyan','fr_isothiocyan',
            'fr_ketone','fr_ketone_Topliss','fr_lactam','fr_lactone','fr_methoxy','fr_morpholine',
            'fr_nitrile','fr_nitro','fr_nitro_arom','fr_nitro_arom_nonortho','fr_nitroso',
            'fr_oxazole','fr_oxime','fr_para_hydroxylation','fr_phenol','fr_phenol_noOrthoHbond',
            'fr_phos_acid','fr_phos_ester','fr_piperdine','fr_piperzine','fr_priamide',
            'fr_prisulfonamd','fr_pyridine','fr_quatN','fr_sulfide','fr_sulfonamd','fr_sulfone','fr_term_acetylene',
            'fr_tetrazole', 'fr_thiazole', 'fr_thiocyan', 'fr_thiophene', 'fr_unbrch_alkane', 'fr_urea',
            'MaxEStateIndex','MinEStateIndex','MaxAbsEStateIndex',
            'MinAbsEStateIndex','NumValenceElectrons','NumRadicalElectrons','MaxPartialCharge','MinPartialCharge',
            'MaxAbsPartialCharge','MinAbsPartialCharge','FpDensityMorgan1',
            'FpDensityMorgan2','FpDensityMorgan3','BalabanJ','BertzCT','Chi0',
            'Chi0n','Chi0v','Chi1','Chi1n','Chi1v','Chi2n','Chi2v','Chi3n',
            'Chi3v','Chi4n','Chi4v','HallKierAlpha','Ipc','Kappa1','Kappa2','Kappa3',
            'LabuteASA','TPSA','PEOE_VSA1','PEOE_VSA2','PEOE_VSA3','PEOE_VSA4','PEOE_VSA5',
            'PEOE_VSA6','PEOE_VSA7','PEOE_VSA8','PEOE_VSA9','PEOE_VSA10','PEOE_VSA11','PEOE_VSA12',
            'PEOE_VSA13','PEOE_VSA14', 'SMR_VSA1','SMR_VSA2','SMR_VSA3','SMR_VSA4','SMR_VSA5',
            'SMR_VSA6','SMR_VSA7','SMR_VSA8','SMR_VSA9','SMR_VSA10','SlogP_VSA1','SlogP_VSA2','SlogP_VSA3','SlogP_VSA4','SlogP_VSA5',
            'SlogP_VSA6','SlogP_VSA7','SlogP_VSA8','SlogP_VSA9','SlogP_VSA10',
            'SlogP_VSA11','SlogP_VSA12','EState_VSA1','EState_VSA2','EState_VSA3','EState_VSA4','EState_VSA5','EState_VSA6',
            'EState_VSA7','EState_VSA8','EState_VSA9','EState_VSA10',
            'EState_VSA11','VSA_EState1','VSA_EState2',
            'VSA_EState3','VSA_EState4','VSA_EState5','VSA_EState6','VSA_EState7', 'VSA_EState8', 'VSA_EState9', 'VSA_EState10'
]

class MainWindow(QMainWindow):
      def __init__(self, *args, **kwargs): 
            super(MainWindow, self).__init__(*args, **kwargs)
      
            self.setWindowTitle("DLG")
            
            tabs = QTabWidget()
            tab1 = QWidget()
            tab2 = QWidget() 
            tabs.addTab(tab1,"Optimize")
            tabs.addTab(tab2,"Generate")

            ######### tab1 #################################################################################
          
            main_big_layout = QHBoxLayout()
            tab1.setLayout(main_big_layout)          

            main_frame = QFrame()
            main_layout = QVBoxLayout()             
            main_frame.setLayout(main_layout)
   
            frame1 = QFrame()          
            layout1 = QHBoxLayout()
            label = QLabel('Number of properties:')
            wid_nDes = QComboBox()
            wid_nDes.addItems(['1','2','3','4','5','6'])
            layout1.addWidget(label)
            layout1.addWidget(wid_nDes)
            frame1.setLayout(layout1)     
            main_layout.addWidget(frame1)       

            frame_alldes = QFrame()
            layout_alldes = QVBoxLayout()
            
            def add_descr():
                for i in reversed(range(layout_alldes.count())): 
                      layout_alldes.itemAt(i).widget().setParent(None)
                n = int(wid_nDes.currentText())
                for i in range(n):
                    frame_i = QFrame()
                    layout_i = QHBoxLayout()
                    des_i_n = QLabel('%s.' %(i+1))
                    des_i = QComboBox()
                    des_i.addItems(des_list)
                    des_i_min = QLineEdit('Min')
                    des_i_max = QLineEdit('Max')
                    des_i_rel = QComboBox()
                    des_i_rel.addItems(['Maximize','Minimize'])
                    des_i_opt = QLineEdit('Opt')
                    layout_i.addWidget(des_i_n)
                    layout_i.addWidget(des_i)
                    layout_i.addWidget(des_i_min)
                    layout_i.addWidget(des_i_max)
                    layout_i.addWidget(des_i_rel)
                    layout_i.addWidget(des_i_opt)
                    frame_i.setLayout(layout_i)
                    layout_alldes.addWidget(frame_i)

            frame_alldes.setLayout(layout_alldes)
            main_layout.addWidget(frame_alldes)
            wid_nDes.activated.connect(add_descr)
            wid_nDes.setCurrentIndex(0)
            add_descr()

            frame_rr = QFrame()
            layout_rr = QHBoxLayout()
            label = QLabel('Starting Model path:')
            wid_model_rr = QLineEdit()
            wid_model_rr.setText('Prior.ckpt')
            wid_butt_file_rr = QPushButton('...')
            layout_rr.addWidget(label)
            layout_rr.addWidget(wid_model_rr)
            layout_rr.addWidget(wid_butt_file_rr)
            frame_rr.setLayout(layout_rr)
            main_layout.addWidget(frame_rr)
          
            def load_agent_rr():
                filename, filter = QFileDialog.getOpenFileName(caption='Open file', directory='.', filter='ckpt Files (*.ckpt)')

                if filename:
                   wid_model_rr.setText(filename)
            
            wid_butt_file_rr.clicked.connect(load_agent_rr)
            

            frame6 = QFrame()
            layout6 = QHBoxLayout()
            label = QLabel('Batch Size:')
            wid_bsize = QLineEdit('100')
            layout6.addWidget(label)
            layout6.addWidget(wid_bsize)
            frame6.setLayout(layout6)
            main_layout.addWidget(frame6)


            #frame2 = QFrame()
            #layout2 = QHBoxLayout()
            #label = QLabel('Learning Rate:')
            #wid_lrate = QLineEdit('0.0005')
            #layout2.addWidget(label)
            #layout2.addWidget(wid_lrate)
            #frame2.setLayout(layout2)
            #main_layout.addWidget(frame2)        

            #frame3 = QFrame()
            #layout3 = QHBoxLayout()
            #label = QLabel('Sigma:')
            #wid_sigma = QLineEdit('20.0')
            #layout3.addWidget(label)
            #layout3.addWidget(wid_sigma)
            #frame3.setLayout(layout3)
            #main_layout.addWidget(frame3)

            #frame4 = QFrame()
            #layout4 = QHBoxLayout()
            #label = QLabel('Experience:')
            #wid_exp = QLineEdit('0')
            #layout4.addWidget(label)
            #layout4.addWidget(wid_exp)
            #frame4.setLayout(layout4)
            #main_layout.addWidget(frame4)

            frame5 = QFrame()
            layout5 = QHBoxLayout()
            label = QLabel('Iterations:')
            wid_niter = QLineEdit('100')
            layout5.addWidget(label)
            layout5.addWidget(wid_niter)
            frame5.setLayout(layout5)  
            main_layout.addWidget(frame5)

            frame7 = QFrame()
            layout7 = QHBoxLayout()
            label = QLabel('Output Dir:')
            wid_out = QLineEdit('prova')
            layout7.addWidget(label)
            layout7.addWidget(wid_out)
            frame7.setLayout(layout7)
            main_layout.addWidget(frame7)
     
            button = QPushButton('Do!')
            main_layout.addWidget(button)
            button_stop = QPushButton('Stop!')
            main_layout.addWidget(button_stop)

            main_big_layout.addWidget(main_frame)

            figure = Figure()
            ax = figure.add_subplot(111)
            canvas = FigureCanvas(figure)
            main_big_layout.addWidget(canvas)
            
            ######### tab2 #################################################################################
            
            main_main_lay_t2 = QHBoxLayout()
            tab2.setLayout(main_main_lay_t2)
            
            main_big_frame_t2 = QFrame()
            main_big_layout_t2 = QVBoxLayout()
            main_big_frame_t2.setLayout(main_big_layout_t2)          

            frame1_t2 = QFrame()
            layout1_t2 = QHBoxLayout()
            label = QLabel('Number of mols to generate:')
            wid_nmols = QLineEdit('100')
            layout1_t2.addWidget(label)
            layout1_t2.addWidget(wid_nmols)
            frame1_t2.setLayout(layout1_t2)
            main_big_layout_t2.addWidget(frame1_t2)
          
            frame2_t2 = QFrame()
            layout2_t2 = QHBoxLayout()
            label = QLabel('Model path:')
            wid_model = QLineEdit()
            wid_butt_file = QPushButton('...')
            layout2_t2.addWidget(label)
            layout2_t2.addWidget(wid_model)
            layout2_t2.addWidget(wid_butt_file)
            frame2_t2.setLayout(layout2_t2)
            main_big_layout_t2.addWidget(frame2_t2)
          
            def load_agent():
                filename, filter = QFileDialog.getOpenFileName(caption='Open file', directory='.', filter='ckpt Files (*.ckpt)')

                if filename:
                   wid_model.setText(filename)
            
            wid_butt_file.clicked.connect(load_agent)
            
            frame3_t2 = QFrame()          
            layout3_t2 = QHBoxLayout()
            label = QLabel('Number of properties:')
            wid_nDes_t2 = QComboBox()
            wid_nDes_t2.addItems(['1','2','3','4','5','6'])
            layout3_t2.addWidget(label)
            layout3_t2.addWidget(wid_nDes_t2)
            frame3_t2.setLayout(layout3_t2)     
            main_big_layout_t2.addWidget(frame3_t2)       

            frame_alldes_t2 = QFrame()
            layout_alldes_t2 = QVBoxLayout()
            
            def add_descr_t2():
                for i in reversed(range(layout_alldes_t2.count())): 
                      layout_alldes_t2.itemAt(i).widget().setParent(None)
                n = int(wid_nDes_t2.currentText())
                for i in range(n):
                    frame_i = QFrame()
                    layout_i = QHBoxLayout()
                    des_i_n = QLabel('%s.' %(i+1))
                    des_i = QComboBox()
                    des_i.addItems(des_list)
                    des_i_min = QLineEdit('Min')
                    des_i_max = QLineEdit('Max')
                    des_i_rel = QComboBox()
                    des_i_rel.addItems(['Maximize','Minimize'])
                    des_i_opt = QLineEdit('Opt')
                    layout_i.addWidget(des_i_n)
                    layout_i.addWidget(des_i)
                    layout_i.addWidget(des_i_min)
                    layout_i.addWidget(des_i_max)
                    layout_i.addWidget(des_i_rel)
                    layout_i.addWidget(des_i_opt)
                    frame_i.setLayout(layout_i)
                    layout_alldes_t2.addWidget(frame_i)
            
            frame_alldes_t2.setLayout(layout_alldes_t2)
            main_big_layout_t2.addWidget(frame_alldes_t2)
            wid_nDes_t2.activated.connect(add_descr_t2)
            wid_nDes_t2.setCurrentIndex(0)
            add_descr_t2()
            
            
            frame4_t2 = QFrame()
            layout4_t2 = QHBoxLayout()
            label = QLabel('Output Dir:')
            wid_out_t2 = QLineEdit('prova')
            layout4_t2.addWidget(label)
            layout4_t2.addWidget(wid_out_t2)
            frame4_t2.setLayout(layout4_t2)
            main_big_layout_t2.addWidget(frame4_t2)
         
            button_gen = QPushButton('Generete!')
            main_big_layout_t2.addWidget(button_gen)
            
            main_main_lay_t2.addWidget(main_big_frame_t2)
            
            
            main_big_frame_t2_right = QFrame()
            main_big_lay_t2_right = QVBoxLayout()
            main_big_frame_t2_right.setLayout(main_big_lay_t2_right)
            
            hist = Figure()
            ax_hist = hist.add_subplot(111)
            canvas_hist = FigureCanvas(hist)
            main_big_lay_t2_right.addWidget(canvas_hist)
            
            frame_des_his = QFrame()     
            layout_des_his = QHBoxLayout()
            label = QLabel('Select descriptor to plot:')
            wid_des_his = QComboBox()
            layout_des_his.addWidget(label)
            layout_des_his.addWidget(wid_des_his)
            frame_des_his.setLayout(layout_des_his)     
            main_big_lay_t2_right.addWidget(frame_des_his)
               
            
            main_main_lay_t2.addWidget(main_big_frame_t2_right)
            
            #########################################################
              
            self.setCentralWidget(tabs)

            def plot_data():
                try:
                   score = np.load(os.path.join(wid_out.text(), "Scores.npy"))
                   length = 50
                   early_cumsum = np.cumsum(score[1][:length]) / np.arange(1, min(len(score[1]), length) + 1)
                   if (len(score[1])>length):
                      cumsum = np.cumsum(score[1])
                      cumsum =  (cumsum[length:] - cumsum[:-length]) / length
                      cumsum = np.concatenate((early_cumsum, cumsum))
                      ravg = cumsum
                   else:   
                      ravg = early_cumsum
                   
                   ax.cla()
                   ax.set_xlabel('Iteration')
                   ax.set_ylabel('Score')
                   ax.plot(score[0],score[1], '-', color='blue', label='Average')
                   ax.plot(np.array(range(len(ravg)))+1,ravg, '-', color='red', label='Running Average')
                   ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins='auto', integer=True, steps = [1, 2, 2.5, 5, 10]))
                   ax.legend(loc='lower right')
                   
                   canvas.draw()
                except:
                   print('calculating')
   
            def obtain_data():
                n = int(wid_nDes.currentText())
                des_names = [] 
                des_mins = []
                des_maxs = []
                des_rels = []
                des_opts = []
                for k in range(n):
                    des_names.append(layout_alldes.itemAt(k).widget().layout().itemAt(1).widget().currentText())
                    des_mins.append(float(layout_alldes.itemAt(k).widget().layout().itemAt(2).widget().text()))
                    des_maxs.append(float(layout_alldes.itemAt(k).widget().layout().itemAt(3).widget().text()))
                    des_rels.append(layout_alldes.itemAt(k).widget().layout().itemAt(4).widget().currentText()) 
                    des_opts.append(layout_alldes.itemAt(k).widget().layout().itemAt(5).widget().text())
                
                start_agent = wid_model_rr.text()
                bsize = int(wid_bsize.text())
                #lrate = float(wid_lrate.text())
                #sigma = float(wid_sigma.text())
                #exp = int(wid_exp.text())
                niter = int(wid_niter.text())
                out = wid_out.text()   
                
                dict_param = {'scoring_function': 'logp_mw', 'scoring_function_kwargs': {'des_names': des_names, 'des_mins': des_mins, 'des_maxs': des_maxs, 'des_rels': des_rels, 'des_opts': des_opts}, 'learning_rate': 0.0005, 'n_steps': niter, 'batch_size': bsize, 'sigma': 20, 'experience_replay': 0, 'num_processes': 0, 'restore_prior_from': 'Prior.ckpt', 'restore_agent_from': start_agent, 'save_dir': out}
                print (dict_param)
                global proc
                proc = mp.Process(target=train_agent, kwargs=(dict_param))
                proc.start()
                global timer
                timer = QTimer()
                timer.timeout.connect(plot_data)
                timer.start(1000)

            def stop_calc():
                proc.terminate()
                timer.stop()
                print ("Stopped")

            def generate_data():
                n = int(wid_nDes_t2.currentText())
                global des_names
                des_names = []
                des_mins = []
                des_maxs = []
                des_rels = []
                des_opts = []
                for k in range(n):
                    des_names.append(layout_alldes_t2.itemAt(k).widget().layout().itemAt(1).widget().currentText())
                    des_mins.append(float(layout_alldes_t2.itemAt(k).widget().layout().itemAt(2).widget().text()))
                    des_maxs.append(float(layout_alldes_t2.itemAt(k).widget().layout().itemAt(3).widget().text()))
                    des_rels.append(layout_alldes_t2.itemAt(k).widget().layout().itemAt(4).widget().currentText())
                    des_opts.append(layout_alldes_t2.itemAt(k).widget().layout().itemAt(5).widget().text())

                nmols = int(wid_nmols.text())
                global out
                out = wid_out_t2.text()
                model = wid_model.text()

                dict_param = {'scoring_function': 'logp_mw', 'scoring_function_kwargs': {'des_names': des_names, 'des_mins': des_mins, 'des_maxs': des_maxs, 'des_rels': des_rels, 'des_opts': des_opts}, 'num_processes': 0, 'restore_agent_from': model, 'save_dir': out, 'batch_size': nmols}
                
                global proc_sample
                proc_sample = mp.Process(target=sample_agent, kwargs=(dict_param))
                proc_sample.start()
                proc_sample.join()
                box_compl = QMessageBox(QMessageBox.Information, "DLG", "Generation Complete !", QMessageBox.Ok)
                box_compl.exec_()
                add_descr_to_hist()
                
            def add_descr_to_hist():
                wid_des_his.clear()     
                des_names.append('Score')
                wid_des_his.addItems(des_names)
                         
            def plot_hist():
                v = []
                with open(os.path.join(out, "sampled.txt"),"r") as fff:
                     for line in fff:
                         v.append(line.split()[1:])
                v = np.array(v).astype('float')
                idx = wid_des_his.currentIndex()
                ax_hist.cla()
                ax_hist.hist(v[:,idx],bins=10)
                ax_hist.xaxis.set_major_locator(ticker.AutoLocator())   
                canvas_hist.draw() 
                   
            button.clicked.connect(obtain_data)
            button_stop.clicked.connect(stop_calc)

            button_gen.clicked.connect(generate_data)
            wid_des_his.currentIndexChanged.connect(plot_hist)

if __name__ == '__main__':
     mp.set_start_method('spawn')
     app = QApplication([])
     window = MainWindow()
     window.show()
     app.exec_()
