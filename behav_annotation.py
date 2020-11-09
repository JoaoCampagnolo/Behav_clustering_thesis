from enum import Enum


class Behav(Enum):
    WALK_FORW = 0
    WALK_BACKW = 1
    PUSH_BALL = 2
    REST = 3
    GROOM_FLEG = 4
    GROOM_ANT = 5
    NONE = 6

# Reassign data directories: new folder is /Volumes/jhc_data/JHC/paper...
label_gt_list = [([0, 140], Behav.REST, 'paper/180919_MDN_CsCh/Fly6/001_SG1'),
                 ((140, 460), Behav.WALK_BACKW, 'paper/180919_MDN_CsCh/Fly6/001_SG1'),
                 ((600, 750), Behav.WALK_FORW, 'paper/180919_MDN_CsCh/Fly6/001_SG1'),
                 ((750, 900), Behav.REST, 'paper/180919_MDN_CsCh/Fly6/001_SG1'),

                 ([0, 140], Behav.REST, 'paper/180919_MDN_CsCh/Fly6/002_SG1'),
                 ((140, 500), Behav.WALK_BACKW, 'paper/180919_MDN_CsCh/Fly6/002_SG1'),
                 ((630, 800), Behav.WALK_FORW, 'paper/180919_MDN_CsCh/Fly6/002_SG1'),
                 ((790, 900), Behav.REST, 'paper/180919_MDN_CsCh/Fly6/002_SG1'),

                 ([0, 140], Behav.REST, 'paper/180919_MDN_CsCh/Fly6/003_SG1'),
                 ((140, 500), Behav.WALK_BACKW, 'paper/180919_MDN_CsCh/Fly6/003_SG1'),
                 ((570, 750), Behav.WALK_FORW, 'paper/180919_MDN_CsCh/Fly6/003_SG1'),

                 ([0, 140], Behav.REST, 'paper/180919_MDN_CsCh/Fly6/004_SG1'),
                 ((140, 500), Behav.WALK_BACKW, 'paper/180919_MDN_CsCh/Fly6/004_SG1'),
                 ((600, 750), Behav.WALK_FORW, 'paper/180919_MDN_CsCh/Fly6/004_SG1'),

                 ([0, 140], Behav.REST, 'paper/180919_MDN_CsCh/Fly6/005_SG1'),
                 ((140, 500), Behav.WALK_BACKW, 'paper/180919_MDN_CsCh/Fly6/005_SG1'),
                 ((600, 750), Behav.WALK_FORW, 'paper/180919_MDN_CsCh/Fly6/005_SG1'),

                 ((0, 150), Behav.GROOM_FLEG, 'paper/180921_aDN_CsCh/Fly6/003_SG1'),
                 ((170, 350), Behav.GROOM_ANT, 'paper/180921_aDN_CsCh/Fly6/003_SG1'),
                 ((450, 600), Behav.REST, 'paper/180921_aDN_CsCh/Fly6/003_SG1'),

                 ((0, 150), Behav.REST, 'paper/180921_aDN_CsCh/Fly6/001_SG1'),
                 ((180, 350), Behav.GROOM_ANT, 'paper/180921_aDN_CsCh/Fly6/001_SG1'),
                 ((400, 580), Behav.REST, 'paper/180921_aDN_CsCh/Fly6/001_SG1'),

                 ((250, 600), Behav.WALK_BACKW, 'paper/180918_MDN_CsCh/Fly2/004_SG1'),

                 ((190, 300), Behav.GROOM_ANT, 'paper/180921_aDN_CsCh/Fly4/003_SG1'),

                 ((400, 900), Behav.WALK_FORW, 'paper/180918_MDN_PR/Fly1/003_SG1'),

                 ((0, 500), Behav.REST, 'paper/180918_MDN_PR/Fly1/004_SG1'),
                 ((650, 900), Behav.WALK_FORW, 'paper/180918_MDN_PR/Fly1/004_SG1'),

                 ((0, 500), Behav.REST, 'paper/180918_MDN_PR/Fly1/005_SG1'),
                 ((500, 900), Behav.WALK_FORW, 'paper/180918_MDN_PR/Fly1/005_SG1'),

                 ((0, 100), Behav.PUSH_BALL, 'paper/180918_MDN_PR/Fly2/001_SG1'),

                 ((350, 500), Behav.GROOM_FLEG, 'paper/180918_MDN_PR/Fly2/002_SG1'),

                 ((400, 530), Behav.GROOM_FLEG, 'paper/180918_MDN_PR/Fly2/003_SG1'),

                 ((150, 230), Behav.GROOM_ANT, 'paper/180921_aDN_CsCh/Fly3/001_SG1'),

                 #((170, 210), Behav.WALK_BACKW, 'paper/180919_MDN_CsCh/Fly4/005_SG1'),
                 #((210, 600), Behav.WALK_FORW, 'paper/180919_MDN_CsCh/Fly4/005_SG1'),
                 #((600, 700), Behav.PUSH_BALL, 'paper/180919_MDN_CsCh/Fly4/005_SG1'),
                 ]

train_dir = ['/Volumes/jhc_data/JHC/paper/180920_aDN_PR/Fly2/001_SG1',
             '/Volumes/jhc_data/JHC/paper/180920_aDN_PR/Fly2/003_SG1',
             '/Volumes/jhc_data/JHC/paper/180920_MDN_PR/Fly9/001_SG1',
             '/Volumes/jhc_data/JHC/paper/180920_MDN_PR/Fly9/005_SG1',
             '/Volumes/jhc_data/JHC/paper/180921_aDN_CsCh/Fly3/001_SG1',
             '/Volumes/jhc_data/JHC/paper/180921_aDN_CsCh/Fly5/001_SG1',
             '/Volumes/jhc_data/JHC/paper/180921_aDN_CsCh/Fly6/003_SG1',
             '/Volumes/jhc_data/JHC/paper/180921_aDN_CsCh/Fly7/003_SG1']

val_dir = ['/Volumes/jhc_data/JHC/paper/180920_aDN_PR/Fly2/002_SG1',
           '/Volumes/jhc_data/JHC/paper/180920_MDN_PR/Fly9/002_SG1',
           '/Volumes/jhc_data/JHC/paper/180921_aDN_CsCh/Fly4/002_SG1',
           '/Volumes/jhc_data/JHC/paper/180921_aDN_CsCh/Fly5/002_SG1',
           '/Volumes/jhc_data/JHC/paper/180921_aDN_CsCh/Fly6/004_SG1',
           '/Volumes/jhc_data/JHC/paper/180921_aDN_CsCh/Fly7/004_SG1']

train_dir_single = ['/Volumes/jhc_data/JHC/paper/180920_aDN_PR/Fly2/001_SG1']

all_dir = ['/Volumes/jhc_data/JHC/paper/180920_aDN_PR/Fly2/001_SG1',
           '/Volumes/jhc_data/JHC/paper/180920_aDN_PR/Fly2/002_SG1', '/Volumes/jhc_data/JHC/paper/180920_aDN_PR/Fly2/003_SG1',
           '/Volumes/jhc_data/JHC/paper/180920_aDN_PR/Fly2/004_SG1', '/Volumes/jhc_data/JHC/paper/180920_aDN_PR/Fly2/005_SG1',
           '/Volumes/jhc_data/JHC/paper/180920_MDN_PR/Fly9/001_SG1', '/Volumes/jhc_data/JHC/paper/180920_MDN_PR/Fly9/002_SG1',
           '/Volumes/jhc_data/JHC/paper/180920_MDN_PR/Fly9/003_SG1', '/Volumes/jhc_data/JHC/paper/180920_MDN_PR/Fly9/004_SG1',
           '/Volumes/jhc_data/JHC/paper/180920_MDN_PR/Fly9/005_SG1',
           '/Volumes/jhc_data/JHC/paper/180921_aDN_CsCh/Fly3/001_SG1',
           '/Volumes/jhc_data/JHC/paper/180921_aDN_CsCh/Fly4/001_SG1', 
           '/Volumes/jhc_data/JHC/paper/180921_aDN_CsCh/Fly4/002_SG1',
           '/Volumes/jhc_data/JHC/paper/180921_aDN_CsCh/Fly4/003_SG1',
           '/Volumes/jhc_data/JHC/paper/180921_aDN_CsCh/Fly4/005_SG1',
           '/Volumes/jhc_data/JHC/paper/180921_aDN_CsCh/Fly5/001_SG1', 
           '/Volumes/jhc_data/JHC/paper/180921_aDN_CsCh/Fly5/002_SG1',
           '/Volumes/jhc_data/JHC/paper/180921_aDN_CsCh/Fly5/003_SG1', 
           '/Volumes/jhc_data/JHC/paper/180921_aDN_CsCh/Fly5/004_SG1',
           '/Volumes/jhc_data/JHC/paper/180921_aDN_CsCh/Fly5/005_SG1', 
           '/Volumes/jhc_data/JHC/paper/180921_aDN_CsCh/Fly6/002_SG1',
           '/Volumes/jhc_data/JHC/paper/180921_aDN_CsCh/Fly6/003_SG1', 
#            '/Volumes/jhc_data/JHC/paper/180921_aDN_CsCh/Fly6/004_SG1', INCOMPLETE VIDEO DATA???
           '/Volumes/jhc_data/JHC/paper/180921_aDN_CsCh/Fly6/005_SG1', 
           '/Volumes/jhc_data/JHC/paper/180921_aDN_CsCh/Fly7/001_SG1',
           '/Volumes/jhc_data/JHC/paper/180921_aDN_CsCh/Fly7/002_SG1', 
           '/Volumes/jhc_data/JHC/paper/180921_aDN_CsCh/Fly7/003_SG1',
           '/Volumes/jhc_data/JHC/paper/180921_aDN_CsCh/Fly7/004_SG1',
           '/Volumes/jhc_data/JHC/paper/180921_aDN_CsCh/Fly7/005_SG1' ]


train_colors = ['red', 'green', 'blue', 'orange', 'purple', 'palegreen', 'sienna', 'magenta']
val_colors = ['yellow', 'cyan', 'crimson', 'darkgrey', 'salmon', 'olive' , 'darkcyan']

fly_tags = ['180920_aDN_PR/Fly2/001_SG1', '180920_aDN_PR/Fly2/002_SG1', '180920_aDN_PR/Fly2/003_SG1',
            '180920_aDN_PR/Fly2/004_SG1', '180920_aDN_PR/Fly2/005_SG1', '180920_MDN_PR/Fly9/001_SG1', 
            '180920_MDN_PR/Fly9/002_SG1', '180920_MDN_PR/Fly9/003_SG1', '180920_MDN_PR/Fly9/004_SG1',
            '180920_MDN_PR/Fly9/005_SG1', '180921_aDN_CsCh/Fly3/001_SG1', '180921_aDN_CsCh/Fly4/001_SG1', 
            '180921_aDN_CsCh/Fly4/002_SG1', '180921_aDN_CsCh/Fly4/003_SG1', '180921_aDN_CsCh/Fly4/005_SG1',
            '180921_aDN_CsCh/Fly5/001_SG1', '180921_aDN_CsCh/Fly5/002_SG1', '180921_aDN_CsCh/Fly5/003_SG1', 
            '180921_aDN_CsCh/Fly5/004_SG1', '180921_aDN_CsCh/Fly5/005_SG1', '180921_aDN_CsCh/Fly6/002_SG1',
            '180921_aDN_CsCh/Fly6/003_SG1', '180921_aDN_CsCh/Fly6/004_SG1', '180921_aDN_CsCh/Fly6/005_SG1', 
            '180921_aDN_CsCh/Fly7/001_SG1', '180921_aDN_CsCh/Fly7/002_SG1', '180921_aDN_CsCh/Fly7/003_SG1',
            '180921_aDN_CsCh/Fly7/004_SG1', '180921_aDN_CsCh/Fly7/005_SG1' ]
            
# Only 1 species flies for testing joint angles
adn_names = ['pose_result__data_paper_180920_aDN_PR_Fly2_001_SG1',
             'pose_result__data_paper_180920_aDN_PR_Fly2_005_SG1',
             'pose_result__data_paper_180921_aDN_PR_Fly4_001_SG1',
             'pose_result__data_paper_180921_aDN_PR_Fly4_005_SG1',
             'pose_result__data_paper_180921_aDN_PR_Fly5_001_SG1',
             'pose_result__data_paper_180921_aDN_PR_Fly5_005_SG1',
             'pose_result__data_paper_180921_aDN_PR_Fly6_001_SG1',
             'pose_result__data_paper_180921_aDN_PR_Fly7_001_SG1',
             'pose_result__data_paper_180921_aDN_PR_Fly7_005_SG1',
             'pose_result__data_paper_180921_aDN_PR_Fly8_001_SG1',
             'pose_result__data_paper_180921_aDN_PR_Fly8_005_SG1',
             'pose_result__data_paper_180921_aDN_PR_Fly9_001_SG1',
             'pose_result__data_paper_180921_aDN_PR_Fly9_005_SG1' ]

adn_save = ['/Users/joaohenrique/Documents/EPFL/joao_pose/pose_result/adn_pr'] #* len(adn_names)
#adn_pr_dir = [f'{a_}{b_}' for a_, b_ in zip(adn_save, adn_names)]

adn_pr_tags = ['180921_aDN_PR_Fly8_001', '180921_aDN_PR_Fly4_001', '180921_aDN_PR_Fly4_005', '180921_aDN_PR_Fly8_005',  '180921_aDN_PR_Fly5_005', '180921_aDN_PR_Fly9_005', '180921_aDN_PR_Fly9_001', '180921_aDN_PR_Fly5_001',  '180921_aDN_PR_Fly7_005', '180920_aDN_PR_Fly2_005', '180920_aDN_PR_Fly2_001', '180921_aDN_PR_Fly7_001',  '180921_aDN_PR_Fly6_001']

adn_val = ['/Users/joaohenrique/Documents/EPFL/joao_pose/pose_result/validation']