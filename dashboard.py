## Importation des library importants 
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
## # import seaborn as sns
import pickle
import time
import shap
import math
from urllib.request import urlopen
import json
import requests
import plotly.graph_objects as go 

from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
## # import display_client_info as display_client_info
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
## # Implémentez un modèle de scoring

###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################
st.markdown("""
<style>
body {
  background: #ff0099; 
  background: -webkit-linear-gradient(to right, #ff0099, #493240); 
  background: linear-gradient(to right, #ff0099, #493240); 
}
</style>
    """, unsafe_allow_html=True)



st.title('Implémentez un modèle de scoring')
##st.image('images/entreprise.png') 


st.sidebar.header("Description générale : ")

html_temp = """
<div style="background-color:brown; padding:10px; border-radius:5px">
<h1 style="color: white; text-align:center"> </h1>
</div>
<p style="font-size: 20px; font-weight: bold; text-align:center">
</p> """


check_box2 = st.sidebar.checkbox(label = 'Description du projet')
if (check_box2):
    st.image('images/entreprise.PNG') 
        
    #st.title('Implémentez un modèle de scoring')
    st.write('''
    **L'objectif du projet :**
    - Construire un modèle de scoring qui donnera une prédiction sur la probabilité de faillite d'un client de façon automatique. 
    - Construire un dashboard interactif à destination des gestionnaires de la relation client permettant d'interpréter les prédictions faites par le modèle et d’améliorer la connaissance client des chargés de relation client.

    **Statut de remboursement du prêt**
    - **la valeur 0** : signifie que le prêt est remboursé
    - **la valeur 1** : signifie que le prêt n'est pas remboursé.

    ''')

    



# st.sidebar.header("Les parametres d'entrées : ")




###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################
def main() :

    @st.cache
    def load_data():
        PATH = 'dataset/'
        #données test après feature engeniering
        df = pd.read_parquet(PATH+'test_df.parquet')
        
        #données test avant feature engeniering
        data_test = pd.read_parquet(PATH+'application_test.parquet')
        
        #données train avant feature engeniering
        data_train = pd.read_parquet(PATH+'application_train.parquet')
        
        #données test avant feature engeniering
        X_test = pd.read_csv(PATH+'X_test.parquet',encoding="ISO-8859-1", on_bad_lines='skip',lineterminator='\n')
        
        #données train avant feature engeniering
        y_test = pd.read_csv(PATH+'y_test.csv')        
        
        #description des features
        description = pd.read_csv(PATH+'HomeCredit_columns_description.csv', 
                                      usecols=['Row', 'Description'], \
                                  index_col=0, encoding='unicode_escape')

        return df, data_test, data_train, y_test , X_test ,  description   # ,X_test 
    
###########################################################################################################################################
###########################################################################################################################################


                      

########################################################################################################################################### 
    @st.cache
    def load_model():
        '''loading the trained model'''
        return pickle.load(open('./model/ModelClassifier.pkl', 'rb'))

    @st.cache
    def get_client_info(data, id_client):
        client_info = data[data['SK_ID_CURR']==int(id_client)]
        return client_info

    #@st.cache
    def plot_distribution(applicationDF,feature, client_feature_val, title):

        if (not (math.isnan(client_feature_val))):
            fig = plt.figure(figsize = (10, 4))

            t0 = applicationDF.loc[applicationDF['TARGET'] == 0]
            t1 = applicationDF.loc[applicationDF['TARGET'] == 1]

            if (feature == "DAYS_BIRTH"):
                sns.kdeplot((t0[feature]/-365).dropna(), label = 'Remboursé', color='g')
                sns.kdeplot((t1[feature]/-365).dropna(), label = 'Défaillant', color='r')
                plt.axvline(float(client_feature_val/-365), \
                            color="blue", linestyle='--', label = 'Position Client')

            elif (feature == "DAYS_EMPLOYED"):
                sns.kdeplot((t0[feature]/365).dropna(), label = 'Remboursé', color='g')
                sns.kdeplot((t1[feature]/365).dropna(), label = 'Défaillant', color='r')    
                plt.axvline(float(client_feature_val/365), color="blue", \
                            linestyle='--', label = 'Position Client')

            else:    
                sns.kdeplot(t0[feature].dropna(), label = 'Remboursé', color='g')
                sns.kdeplot(t1[feature].dropna(), label = 'Défaillant', color='r')
                plt.axvline(float(client_feature_val), color="blue", \
                            linestyle='--', label = 'Position Client')


            plt.title(title, fontsize='20', fontweight='bold')
            #plt.ylabel('Nombre de clients')
            #plt.xlabel(fontsize='14')
            plt.legend()
            plt.show()  
            st.pyplot(fig)
        else:
            st.write("Comparaison impossible car la valeur de cette variable n'est pas renseignée (NaN)")
            
###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################
    #@st.cache
    def univariate_categorical(applicationDF,feature,client_feature_val,\
                               titre,ylog=False,label_rotation=False,
                               horizontal_layout=True):
        if (client_feature_val.iloc[0] != np.nan):

            temp = applicationDF[feature].value_counts()
            df1 = pd.DataFrame({feature: temp.index,'Number of contracts': temp.values})

            categories = applicationDF[feature].unique()
            categories = list(categories)

            # Calculate the percentage of target=1 per category value
            cat_perc = applicationDF[[feature,\
                                      'TARGET']].groupby([feature],as_index=False).mean()
            cat_perc["TARGET"] = cat_perc["TARGET"]*100
            cat_perc.sort_values(by='TARGET', ascending=False, inplace=True)

            ####if(horizontal_layout):
            ####    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))
            ####else:
            ####    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(20,24))

            # 1. Subplot 1: Count plot of categorical column
            # sns.set_palette("Set2")
            fig1 = px.histogram(applicationDF, x=feature, color = 'NAME_CONTRACT_TYPE' ,) #  barmode='group' ,
            fig1.update_layout(title_text='TYPE DE CONTRAT', title_x=0.4)
            fig1.update_layout(legend_traceorder="reversed")
            fig1.show()

            # If the plot is not readable, use the log scale.
            ####if ylog:
            ####    ax1.set_yscale('log')
            ####    ax1.set_ylabel("Count (log)",fontdict={'fontsize' : 15, \
            ####                                           'fontweight' : 'bold'})   
            ####if(label_rotation):
            ####    s.set_xticklabels(s.get_xticklabels(),rotation=90)

            #####################################################################3
            # 2. Subplot 2: Percentage of defaulters within the categorical column
            df = applicationDF.groupby(by=["TARGET", feature]).size().reset_index(name="counts")
            #df = df.groupby(by=["Name", "Defect severity"]).size().reset_index(name="counts")
            fig2 = px.bar(data_frame=df, x="TARGET", y="counts", #color = 'counts' , 
            color=feature, barmode="group",
            opacity=0.8, orientation='v', title='TYPE DE CONTRAT',
            labels={'x': 'Type de contrat', 'y':'Count'}
            #color="NAME_CONTRACT_TYPE",# legend = ['Remboursé','Défaillant'],
            # labels={"sex": "Gender", "smoker": "Smokes"},
            #base=[0,10 , 20 , 50], error_y=[5,10 , 15 , 20], 
            )
            # order of legend is reversed
            fig2.update_layout(title_text='TYPE DE CONTRAT', title_x=0.4)
            fig2.update_layout(legend_traceorder="reversed")
            fig2.show()
            st.plotly_chart(fig1 , fig2 , use_container_width=True)
            #st.pyplot(fig1)
        else:
            st.write("Comparaison impossible car la valeur de cette variable n'est pas renseignée (NaN)")
            
###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################

    #Chargement des données    
    df, data_test, data_train,X_test , y_test ,  description = load_data()

    ignore_features = ['Unnamed: 0','SK_ID_CURR', 'INDEX', 'TARGET']
    relevant_features = [col for col in df if col not in ignore_features]

    #Chargement du modèle
    model = load_model()
    #df
###########################################################################################################################################
###########################################################################################################################################
########################################################################################################################################### 
    ###sampledf = df.sample(100, random_state=42)
    ###st.dataframe(sampledf, width=10, height=10) 
    #df  
    ### Data
    def show_data():
        st.write(data_test.head(15))

    check_box1 = st.sidebar.checkbox(label = 'Echantillon de jeux de données')
    #if check_box1:
    #    show_data()
        

    
    
    
    def display_confusion_model():
        y_proba = model.predict_proba(X_test)
        roc_auc = round(roc_auc_score(y_test, y_proba[:,1]),3)
        cf_matrix_roc_auc(model, y_test, model.predict(X_test), model.predict_proba(X_test)[:,1], roc_auc, "LGBM (Balancing Method: Balanced)")
 
    
    #######################################
    # SIDEBAR
    #######################################

    # LOGO_IMAGE = "logo.png"
    SHAP_GENERAL = "global_feature_importance.png"
    # shap_general = "feature_importance.png"

    with st.sidebar:
        ## st.header(" Prêt à dépenser")

        st.write("## Identificateur du Client")
        #st.write("**ID Client est  :**", id_client)
        id_list = df["SK_ID_CURR"].tolist()
        # id_client = st.number_input("Sélectionner l'identifiant du client" , step = 1 , value = 100001 )
        id_client = st.number_input(" " , step = 1 , value = 100001 )
        # id_client = st.selectbox(
         #    "Sélectionner l'identifiant du client", id_list)

        st.write("## Choisir une opération")
        #st.write("**ID Client est  :**", id_client)
        # .sidebar.radio
        #    ------------------------------------------------------------------------
        #    ------------------------------------------------------------------------        
        show_client_details = st.checkbox("Informations fondamentales du client" , value = False)
        ##show_client_suplemntaryinfo = st.checkbox("les informations supplémentaires")
        show_credit_decision = st.checkbox("Décision de crédit")
        #show_credit_model = st.checkbox("Modèle de décision")
        #Evaluation_metric = st.checkbox("Métriques d'évaluation")
        show_metric_model = st.checkbox("Etude comparative Aux autres Clients") 
        # show_client_comparison = st.checkbox("Comparer aux autres clients")
        shap_general = st.checkbox("Features importantes globale")
        if(st.sidebar.checkbox("Description des features")):
            list_features = description.index.to_list()
            list_features = list(dict.fromkeys(list_features))
            feature = st.sidebar.selectbox('Sélectionner une variable',\
                                   sorted(list_features))
            
            desc = description['Description'].loc[description.index == feature][:1]
            st.markdown('**{}**'.format(desc.iloc[0]))
        #    ------------------------------------------------------------------------ 
        #    ------------------------------------------------------------------------
                ### Sidebar
        ### st.sidebar.title("Menus")
       ###  sidebar_selection = st.sidebar.radio(
        ### 'Select Menu:',
        ### ['Overview', 'Data Analysis', 'Model & Prediction','Prédire solvabilité client'],
        #    ------------------------------------------------------------------------
        #show_client_details = st..sidebar.radio("Afficher les informations du client")
        #show_credit_decision = st..sidebar.radio("Afficher la décision de crédit")
        #show_client_comparison = st..sidebar.radio("Comparer aux autres clients")
        #shap_general = st..sidebar.radio("Afficher la feature importance globale")
        #    ------------------------------------------------------------------------
        #    ------------------------------------------------------------------------


            

###########################################################################################################################################
###########################################################################################################################################
### Solvency
    def pie_chart(thres):
        #st.write(100* (data['TARGET']>thres).sum()/data.shape[0])
        percent_sup_seuil =100* (df['TARGET']>thres).sum()/df.shape[0]
        percent_inf_seuil = 100-percent_sup_seuil
        d = {'col1': [percent_sup_seuil,percent_inf_seuil], 'col2': ['% Non Solvable','% Solvable',]}
        df = pd.DataFrame(data=d)
        fig = px.pie(df,values='col1', names='col2', title=' Pourcentage de solvabilité des clients di dataset')
        st.plotly_chart(fig)
    


    ## seuil_risque = st.sidebar.slider("Seuil de Résolution", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    def show_overview():
        st.title("Risque")
        risque_threshold = st.slider(label = 'Seuil de risque', min_value = 0.0,
                        max_value = 1.0 ,
                         value = 0.5,
                         step = 0.1)
        #st.write(risque_threshold)
        pie_chart(risque_threshold) 
    ##show_overview()
###########################################################################################################################################

    #Titre principal

    html_temp = """
    <div style="background-color:brown; padding:10px; border-radius:5px">
    <h1 style="color: white; text-align:center"> </h1>
    </div>
    <p style="font-size: 20px; font-weight: bold; text-align:center"></p>
    """
            
    st.markdown(html_temp, unsafe_allow_html=True)

####    with st.expander("🤔 A quoi sert cette application ?"):
####        st.write("Ce dashboard interactif à destination des gestionnaires de la relation client de l'entreprise **Prêt à dépenser** permet de comprendre et interpréter les décisions potentielles (prédictions faites par un modèle d'apprentissage) d'ottroi ou non de crédit aux clients") 
####        st.text('\n') 
####        st.write("**Objectif**:  répondre au soucis de transparence vis-à-vis des décisions d’octroi de crédit qui va tout à fait dans le sens des valeurs que l’entreprise veut incarner")
        ## st.image(LOGO_IMAGE)


    #Afficher l'ID Client sélectionné
    #st.write("**ID Client est  :**", id_client)

    if (int(id_client) in id_list):
        st.markdown(" ✅ ✅  **Client existe  dans la base de donnée** ✅ ✅ ")

        client_info = get_client_info(data_test, id_client)


########################################################################################################################################### 
########################################################################################################################################### 
###########################################################################################################################################    

        #-------------------------------------------------------
        # Afficher les informations du client
        #-------------------------------------------------------

        personal_info_cols = {
            'CODE_GENDER': "GENRE",
            'DAYS_BIRTH': "AGE",
            'NAME_FAMILY_STATUS': "STATUT FAMILIAL",
            'CNT_CHILDREN': "NB ENFANTS",
            'FLAG_OWN_CAR': "POSSESSION VEHICULE",
            'FLAG_OWN_REALTY': "POSSESSION BIEN IMMOBILIER",
            'NAME_EDUCATION_TYPE': "NIVEAU EDUCATION",
            'OCCUPATION_TYPE': "EMPLOI",
            'DAYS_EMPLOYED': "NB ANNEES EMPLOI",
            'AMT_INCOME_TOTAL': "REVENUS",
            'AMT_CREDIT': "MONTANT CREDIT", 
            'NAME_CONTRACT_TYPE': "TYPE DE CONTRAT",
            'AMT_ANNUITY': "MONTANT ANNUITES",
            'NAME_INCOME_TYPE': "TYPE REVENUS",
            'EXT_SOURCE_1': "EXT_SOURCE_1",
            'EXT_SOURCE_2': "EXT_SOURCE_2",
            'EXT_SOURCE_3': "EXT_SOURCE_3",
            
            'AMT_GOODS_PRICE'         : "Prix des biens",
          ##  'NAME_TYPE_SUITE'         :  "Accompagnateur du client",
          ##  'NAME_EDUCATION_TYPE'     :   "Niveau de scolarité le plus élevé du client",
            'NAME_HOUSING_TYPE'       :   "Situation de logement du client",
            'DAYS_REGISTRATION'       :   "Nbr jours avant la demande de modification de l'inscription",
            'OWN_CAR_AGE'             :   "Age de la voiture",
          ##  'FLAG_MOBIL'              :   "fourni un téléphone portable",
            'OCCUPATION_TYPE'         :   "Occupation",
            'CNT_FAM_MEMBERS'         :   "nombre de membre de la famille",
            'REGION_RATING_CLIENT'    :    "Notre évaluation de la région",
        }
    ## display_client_info(str(client['SK_ID_CURR'].values[0]),str(client['AMT_INCOME_TOTAL'].values[0]),
     ##                str(round(client['DAYS_BIRTH'].values[0])),str(round(client['DAYS_EMPLOYED']/-365).values[0])) 
        
        
        default_list=\
        ["GENRE","AGE","STATUT FAMILIAL","NB ENFANTS","NB ANNEES EMPLOI","NIVEAU EDUCATION","Prix des biens",
      ##   "Accompagnateur du client","Niveau de scolarité le plus élevé du client","Situation de logement du client",
       ##  "Nbr jours avant la demande de modification de l'inscription","Age de la voiture","fourni un téléphone portable",
         "Occupation","nombre de membre de la famille","Notre évaluation de la région",
        ]
        numerical_features = ['DAYS_BIRTH', 'CNT_CHILDREN', 'DAYS_EMPLOYED', 'OWN_CAR_AGE', 'DAYS_REGISTRATION' , 'AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3','NAME_HOUSING_TYPE']

        rotate_label = ["NAME_FAMILY_STATUS", "NAME_EDUCATION_TYPE"]
        horizontal_layout = ["OCCUPATION_TYPE", "NAME_INCOME_TYPE"]
        if check_box1:
            show_data()
        if (show_client_details):
            st.header('‍🧑 Informations démografiques du client')

            with st.spinner('Chargement des informations relatives au client...'):
                personal_info_df = client_info[list(personal_info_cols.keys())]
                #personal_info_df['SK_ID_CURR'] = client_info['SK_ID_CURR']
                personal_info_df.rename(columns=personal_info_cols, inplace=True)

                personal_info_df["AGE"] = int(round(personal_info_df["AGE"]/365*(-1)))
                personal_info_df["NB ANNEES EMPLOI"] = \
                int(round(personal_info_df["NB ANNEES EMPLOI"]/365*(-1)))


                filtered = st.multiselect("Choisir les informations à afficher", \
                                          options=list(personal_info_df.columns),\
                                          default=list(default_list))
                df_info = personal_info_df[filtered] 
                df_info['SK_ID_CURR'] = client_info['SK_ID_CURR']
                df_info = df_info.set_index('SK_ID_CURR')

                st.table(df_info.astype(str).T)
            
            
                show_all_info = st.checkbox("le reste des informations du Client")
            if (show_all_info):
                ## dataframe = dataframe.T
                st.dataframe(client_info)
                    
                    
                    

###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################
     ###    supplemntary_info_cols = {
### 
   ###          'AMT_GOODS_PRICE'         : "Prix des biens",
   ###          'NAME_TYPE_SUITE'         :  "Accompagnateur du client",
   ###          'NAME_EDUCATION_TYPE'     :   "Niveau de scolarité le plus élevé du client",
   ###          'NAME_HOUSING_TYPE'       :   "Situation de logement du client",
    #### ##         'DAYS_REGISTRATION'       :   "Nbr jours avant la demande de modification de l'inscription",
     ###        'OWN_CAR_AGE'             :   "Age de la voiture",
     ###        'FLAG_MOBIL'              :   "fourni un téléphone portable",
     ###        'OCCUPATION_TYPE'         :   "Occupation",
     ###        'CNT_FAM_MEMBERS'         :   "nombre de membre de la famille",
    ###         'REGION_RATING_CLIENT'    :    "Notre évaluation de la région",

       # }
        #-------------------------------------------------------
        # Afficher les informations supplémentaires du client
        #-------------------------------------------------------
    ###     default_list1=\
   ###      ["Prix des biens","Accompagnateur du client","Niveau de scolarité le plus élevé du client","Situation de logement du ### ### client","DAYS_REGISTRATION","Nbr jours avant la demande de modification de l'inscription","Age de la voiture",
 ###         "fourni un téléphone portable", "Occupation","nombre de membre de la famille","Notre évaluation de la région"]
    ###     numerical_features = ['AMT_GOODS_PRICE', 'NAME_TYPE_SUITE', 'NAME_EDUCATION_TYPE', ### 'NAME_HOUSING_TYPE','DAYS_REGISTRATION','OWN_CAR_AGE','FLAG_MOBIL','OCCUPATION_TYPE','CNT_FAM_MEMBERS','REGION_RATING_CLIENT']

        #rotate_label = ["NAME_FAMILY_STATUS", "NAME_EDUCATION_TYPE"]
        #horizontal_layout = ["OCCUPATION_TYPE", "NAME_INCOME_TYPE"]

      ###   if (show_client_suplemntaryinfo):
       ###       st.header('‍🧑 Informations Supplémentaires du client')


      ###   with st.spinner('Chargement des informations supplémentaires du client...'):
        ###     supplemntary_info_df = client_info[list(supplemntary_info_cols.keys())]
         ###        #personal_info_df['SK_ID_CURR'] = client_info['SK_ID_CURR']
         ###    supplemntary_info_df.rename(columns=supplemntary_info_cols, inplace=True)
                
                
         ###    filtereds = st.multiselect("Choisir les informations à afficher", \
          ###                                 options=list(supplemntary_info_df.columns),\
           ###                                default=list(default_list1))
         ###    df_infos = supplemntary_info_df[filtereds] 
         ###    df_infos['SK_ID_CURR'] = client_info['SK_ID_CURR']
          ###   df_infos = df_infos.set_index('SK_ID_CURR')

        ###     st.table(df_infos.astype(str).T)
            
            
            
          ###   show_all_info = st.checkbox("le reste des informations du Client")
        ### if (show_all_info):
            ### dataframe = dataframe.T
          ###   st.dataframe(client_info)
            
            
            
            
            
            
            
            
            
            # supplemntary_info_df["AGE"] = int(round(supplemntary_info_df["AGE"]/365*(-1)))
            # supplemntary_info_df["NB ANNEES EMPLOI"] = \
            # int(round(supplemntary_info_df["NB ANNEES EMPLOI"]/365*(-1)))
         ## ##if (show_all_info):
            ## dataframe = dataframe.T
            ## st.dataframe(client_info)           


            # filtered = st.multiselect("Choisir les informations à afficher", \
            #                               options=list(supplemntary_info_df.columns),\
           #                                default=list(default_list))
            # df_info = supplemntary_info_df[filtered] 
           #  df_info['SK_ID_CURR'] = client_info['SK_ID_CURR']
            # df_info = df_info.set_index('SK_ID_CURR')

           #  st.table(df_info.astype(str).T)
            # show_all_info = st\
           #  .checkbox("le reste des informations du Client")
           #  if (show_all_info):
            #     ## dataframe = dataframe.T
            #     st.dataframe(client_info)

            
            
            
            
            
            
            

            
            
        
            
            
            
###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################

        #-------------------------------------------------------
        # Afficher la décision de crédit
        #-------------------------------------------------------

        if (show_credit_decision):
            st.header('‍ Scoring et décision du modèle')
            risque_threshold = st.slider(label = 'Seuil de risque', min_value = 0.0,
                                         max_value = 1.0 ,
                                         value = 0.5,
                                          step = 0.1)
            st.write(' L seuil est : ',risque_threshold)
            seuil_risque = st.sidebar.slider("Seuil de Résolution", 
                                             min_value=0.0, 
                                             max_value=1.0, 
                                             value=0.5, 
                                             step=0.01)
            
            #st.write(' La seuille est : ',risque_threshold)
            ##########################################################################################
            # df_sample2 = df.sample(100)
            # arr = np.random.normal(1, 1, size=100)
            # # figure2, ax = plt.subplots()
            # ax.hist(df_sample2, bins=20)
            # st.pyplot(figure2)
            #st.area_chart(df_sample2 )##############################
            ##########################################################################################
            #Appel de l'API : 

            #API_url = "http://127.0.0.1:5000/credit/" + str(id_client)
            API_url = "https://heroku-api-model-scoring-ds.herokuapp.com/credit/"+ str(id_client)
            
            #API_url = "https://heroku-api-model-scoring-ds.herokuapp.com/prediction_credit"+ str(id_client)
            #API_url = "http://127.0.0.1:5000/credit/" + str(id_client)
            ##API_url = "https://milouden-api-scoring-model-app-rklb1w.streamlitapp.com/prediction_credit" + str(id_client)  # credit/"
           
            
            
            with st.spinner('Chargement du score du client...'):
                json_url = urlopen(API_url)

                API_data = json.loads(json_url.read())
                classe_predite = API_data['prediction']
                

                proba = 1-API_data['proba']
                client_score = round(proba*100, 2)
            
            
                figure = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    title = {'text': 'Taux de risque de défaut'},
                    value = client_score,
                    number={"prefix": " "},
                    # delta={"position": "top"  },
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    gauge = {'axis': {'range': [None, 100]},
                             'steps' : [
                                 {'range': [0, 10], 'color': "white"},
                                 {'range': [10, 20], 'color': "gray"},
                                 {'range': [20, 30], 'color': "blue"},
                                 {'range': [30, 40], 'color': "magenta"},
                                 {'range': [40, 50], 'color': "cyan"},
                                 {'range': [50, 60], 'color': "yellow"},
                                 {'range': [60, 70], 'color': "green"},
                                 {'range': [70, 80], 'color': "brown"},
                                 {'range': [80, 90], 'color': "red"},
                                 {'range': [90, 100], 'color': "black"},
                                 ],
                             'threshold': {
                            'line': {'color': "black", 'width': 14},
                            'thickness': 0.2,
                            'value': client_score},

                             'bar': {'color': "black", 'thickness' : 0.1},
                            },
                    ))
                    
                figure.update_layout(paper_bgcolor = "lightgray")
                
                #fig.update_layout(width=450, height=250, 
                #                    margin=dict(l=50, r=50, b=0, t=0, pad=4))

                
                decision_final = st.checkbox(" Afficher la décision : ")
                
                # decision_final
                
                ##def decision_final():
                if classe_predite ==1 :
                    #st.write('decision = )
                    decision = '❌ Crédit Refusé'
                else:
                    decision = '✅ Crédit Accordé'                    
                    
                    
                proba = 1-API_data['proba']
                client_score = round(proba*100, 2)

                left_column, right_column = st.columns((1, 2))
                
                                ## left_column.markdown('Risque de défaut: **{}%**'.format(str(client_score)))
                ## left_column.markdown('Seuil par défaut du modèle: **50%**')

                if decision_final == 1:
                    left_column.markdown(
                        'Décision: <span style="color:red">**{}**</span>'.format(decision),\
                        unsafe_allow_html=True)   
                else:    
                    left_column.markdown(
                        'Décision: <span style="color:green">**{}**</span>'\
                        .format(decision), \
                        unsafe_allow_html=True)
                right_column.plotly_chart(figure)
                

            show_local_feature_importance = st.checkbox(
                "Afficher les variables ayant le plus contribué à la décision du modèle ?")
            if (show_local_feature_importance):
                shap.initjs()
                
                #liste_features = np.linspace(1,800,800)
                #liste_features = pd.DataFrame(liste_features)
                #liste_features = liste_features.astype('int')
                #id_client = st.selectbox(
                #    "Sélectionner l'identifiant du client", id_list
                    
                    
                    #number = st.number_input('Sélectionner le nombre de feautures à afficher ?',liste_features)
                number = st.slider('Sélectionner le nombre de feautures à afficher ?', \
                                  2, 50, 8)

                X = df[df['SK_ID_CURR']==int(id_client)]
                X = X[relevant_features]

                fig, ax = plt.subplots(figsize=(15, 15))
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X)
                shap.summary_plot(shap_values[0], X, plot_type ="bar", \
                                  max_display=number, color_bar=False, plot_size=(8, 8))


                st.pyplot(fig)

###########################################################################################################################################
###########################################################################################################################################
########################################################################################################################################### 
          
                
                
                def show_client_prediction():
                    st.subheader("Selectionner source des données du client")
                    selected_choice = st.radio("",('Client existant dans le dataset','Nouveau client'))

                    if selected_choice == 'Client existant dans le dataset':
                        client_id = st.number_input("Donnez Id du Client",100002)
                    if st.button('Prédire Client'):
                        y_pred,y_proba = predict_client_par_ID("randomForest",client_id)
                        st.info('Probabilité de solvabilité du client : '+str(100*y_proba[0][0])+' %')
                        st.info("Notez que 100% => Client non slovable ")

                    if(y_proba[0][0]<seuil_risque):
                        st.success('Client prédit comme solvable')
                    if(y_proba[0][0]>=seuil_risque):
                        st.error('Client prédit comme non solvable !')

                    if selected_choice == 'Nouveau client':   
                        filename = file_selector()
                        st.write('Fichier du nouveau client selectionné `%s`' % filename)
        
                    if st.button('Prédire Client'):
                        nouveau_client = pd.read_csv(filename)
                        y_pred,y_proba = predict_client("randomForest",nouveau_client)
                        st.info('Probabilité de solvabilité du client : '+str(100*y_proba[0][0])+' %')
                        st.info("Notez que 100% => Client non slovable ")
            
                    if(y_proba[0][0]<seuil_risque):
                        st.success('Client prédit comme solvable')
                    if(y_proba[0][0]>=seuil_risque):
                        st.error('Client prédit comme non solvable !')


                    ### Title
                    st.title('Home Credit Default Risk')
            
            
###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################      

        

        #-------------------------------------------------------
        # Comparer le client sélectionné à d'autres clients
        #-------------------------------------------------------

        if (show_metric_model):
            st.header('‍👀 Comparaison aux autres clients')
            #st.subheader("Comparaison avec l'ensemble des clients")
            with st.expander("🔍 Explication de la comparaison faite"):
                st.write("Lorsqu'une variable est sélectionnée, un graphique montrant la distribution de cette variable selon la classe (remboursé ou défaillant) sur l'ensemble des clients (dont on connait l'état de remboursement de crédit) est affiché avec une matérialisation du positionnement du client actuel.") 

            with st.spinner('Chargement de la comparaison liée à la variable sélectionnée'):
                var = st.selectbox("Sélectionner une variable",\
                                   list(personal_info_cols.values()))
                feature = list(personal_info_cols.keys())\
                [list(personal_info_cols.values()).index(var)]    

                if (feature in numerical_features):                
                    plot_distribution(data_train, feature, client_info[feature], var)   
                elif (feature in rotate_label):
                    univariate_categorical(data_train, feature, \
                                           client_info[feature], var, False, True)
                elif (feature in horizontal_layout):
                    univariate_categorical(data_train, feature, \
                                           client_info[feature], var, False, True, True)
                else:
                    univariate_categorical(data_train, feature, client_info[feature], var)

        #-------------------------------------------------------
        # Comparer le client sélectionné à d'autres clients
        #-------------------------------------------------------

 
                                    
                if(st.checkbox("Afficher les clients similaires")):
                    X_test = df
                    # Median imputation of missing values
                    imputer = SimpleImputer(missing_values=np.nan, strategy='median', verbose=0)
                    X_test[X_test==np.inf] = np.nan
                    imputer.fit(X_test)
                    X_test_preproc = imputer.transform(X_test)
                    nbrs = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(X_test_preproc)
                    # On récupère l'indice des plus proches voisins du client
                    
                    
                    indices = nbrs.kneighbors(X_test_preproc[0:1])[1].flatten()
                    st.dataframe(data_test.iloc[indices])
                    

                #-------------------------------------------------------
                # Afficher la feature importance globale
                #-------------------------------------------------------
        

                
###########################################################################################################################################
###########################################################################################################################################
########################################################################################################################################### 


                ### Solvency
                def pie_chart(thres):
                    #st.write(100* (data['TARGET']>thres).sum()/data.shape[0])
                    percent_sup_seuil =100* (data['TARGET']>thres).sum()/data.shape[0]
                    percent_inf_seuil = 100-percent_sup_seuil
                    d = {'col1': [percent_sup_seuil,percent_inf_seuil], 'col2': ['% Non Solvable','% Solvable',]}
                    df = pd.DataFrame(data=d)
                    fig = px.pie(df,values='col1', names='col2', title=' Pourcentage de solvabilité des clients di dataset')
                    st.plotly_chart(fig)

    
    
            
###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################     
#################                def show_model_analysis():
#################                    conf_mtx = matrix_confusion (y_pred_test_export['y_test'],y_pred_test_export['y_predicted'])
#################                        #st.write(conf_mtx)
#################                fig = go.Figure(data=go.Heatmap(
#################                       z=conf_mtx,
#################                        x=[ 'Actual Negative:0','Actual Positive:1'],
#################                       y=['Predict Negative:0','Predict Positive:1'],
#################                       hoverongaps = False))
#################                st.plotly_chart(fig)
#################
#################                fpr, tpr, thresholds = roc_curve(y_pred_test_export['y_test'],y_pred_test_export['y_probability'])

#################                fig = px.area(
#################                    x=fpr, y=tpr,
#################                    title=f'ROC Curve (AUC={auc(fpr, tpr):.4f})',
#################                    labels=dict(x='False Positive Rate', y='True Positive Rate'),
#################                    width=700, height=500
#################                            )
#################                fig.add_shape(
#################                    type='line', line=dict(dash='dash'),
#################                    x0=0, x1=1, y0=0, y1=1
#################                                )
#################
#################                fig.update_yaxes(scaleanchor="x", scaleratio=1)
#################                fig.update_xaxes(constrain='domain')
#################                st.plotly_chart(fig)
                
                
###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################

    

        if (shap_general):
            #st.header('‍Feature importance globale')
            #st.image('feature_importance.png')
            st.title("Liste des features")
            sidebar_selection = st.radio(
                'Séléctionner parametre:',
                ['feature_importance', 'beeswarm', 'violin'],
                )
            if sidebar_selection == 'feature_importance':
                st.header('‍Feature importance globale')
                st.image('images/feature_importance.png')
            if sidebar_selection == 'beeswarm':
                st.header('‍beeswarm')
                st.image('images/beeswarm.png')
            if sidebar_selection == 'violin':
                st.header('‍violin')
                st.image('images/violin.png')   
            
        #if (show_credit_model):
        #    st.header('‍Home Credit Default Risk')
        #    st.image('images/modele_de_prediction.png')  
        
        
    #####    def show_model_cerdit():
    #####        y_proba = model.predict_proba(X_test)
    #####        roc_auc = round(roc_auc_score(y_test, y_proba[:,1]),3)
    #####        cf_matrix_roc_auc(model, y_test, model.predict(X_test), model.predict_proba(X_test)[:,1], roc_auc, "LGBM (Balancing Method: Balanced)")
###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################  ###########################################################################################################################################
###########################################################################################################################################
########################################################################################################################################### 
###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################
        ######if (show_credit_model):
        ######    st.title("Matrice de confusion et modèle")
        ######    sidebar_selection = st.radio(
        ######        'Séléctionner le modèle:',
        ######        ['Dummyclassifier', 'LGBM', 'RandomForest','Logistic_regression'],
        ######        )
        ######    if sidebar_selection == 'Dummyclassifier':
        ######        st.header('‍Dummyclassifier')
        ######        st.image('images/Dummyclassifier.png')
        ######    if sidebar_selection == 'LGBM':
        ######        st.header('‍LGBM')
        ######        st.image('images/LGBM.png')
            
###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################


        ######if (Evaluation_metric):
        ######    st.title("Liste des métriques")
        ######    sidebar_selection = st.radio(
        ######        'Séléctionner parametre:',
        ######        ['AUC_test', 'F2_Score', 'Temps_execution','Recall'],
        ######        )
        ######    if sidebar_selection == 'AUC_test':
        ######        st.header('‍AUC_TEST')
        ######        st.image('images/AUC_TEST.png')
        ######    if sidebar_selection == 'F2_Score':
        ######        st.header('‍F2_Score')
        ######        st.image('images/F2_Score.png')
        ######    if sidebar_selection == 'Temps_execution':
        ######        st.header('‍Temps_execution')
        ######        st.image('images/Temps_execution.png')
        ######    if sidebar_selection == 'Recall':
        ######        st.header('‍Recall')
        ######        st.image('images/Recall.png')            
          ##   st.header('‍AUC_TEST')
          ##   st.image('images/AUC_TEST.png')
        
        ## if (show_metric_model):
        ##    st.header('‍Répartition des status de crédit')
         ##   st.image('images/Répartition des status de crédit.png')


            
            
    else:    
        st.markdown("**❌ ❌ ❌ ❌ ❌ Client n'existe pas dans la base de donnée ❌ ❌ ❌ ❌**")

if __name__ == '__main__':
    main()           

###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################

