import streamlit as st 
import streamlit .components .v1 as components 
import pandas as pd 
import numpy as np 
import matplotlib .pyplot as plt 
import seaborn as sns 
import torch 
import torch .nn .functional as F 
from transformers import AutoModelForSequenceClassification ,AutoTokenizer 
import textattack 
from captum .attr import LayerIntegratedGradients 
from captum .attr import visualization as viz 
from lime .lime_text import LimeTextExplainer 
import shap 
import nltk 
from nltk .tokenize import word_tokenize 


st .set_page_config (
page_title ="Laboratoire Sécurité NLP - DistilBERT",
layout ="wide",
initial_sidebar_state ="expanded"
)


st .markdown ("""
<style>
    h1, h2, h3 { font-family: 'Segoe UI', sans-serif; color: #E6E6E6; padding-bottom: 5px; }
    h1 { border-bottom: 2px solid #4B8BBE; margin-bottom: 25px; }
    h2 { border-bottom: 1px solid #333333; margin-top: 30px; margin-bottom: 20px; color: #4B8BBE;}
    .custom-box { background-color: #1A1C24; padding: 20px; border-radius: 6px; margin-bottom: 20px; font-size: 1.05rem; }
    .box-insight { border-left: 4px solid #FFD43B; } 
    .box-error { border-left: 4px solid #FF4B4B; }   
    .box-neutral { border-left: 4px solid #4B8BBE; } 
    .text-pos { color: #4CAF50; font-weight: 600; }
    .text-neg { color: #FF4B4B; font-weight: 600; }
    div[data-testid="metric-container"] { background-color: #1A1C24; border: 1px solid #2B2F3A; padding: 15px; border-radius: 6px; }
    .stButton>button { width: 100%; border-radius: 4px; font-weight: bold; }
</style>
""",unsafe_allow_html =True )


if 'current_text'not in st .session_state :
    st .session_state .current_text =None 
if 'original_text'not in st .session_state :
    st .session_state .original_text =None 




@st .cache_resource 
def load_ai_models ():
    model_name ="distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer =AutoTokenizer .from_pretrained (model_name )
    model =AutoModelForSequenceClassification .from_pretrained (model_name )
    device =torch .device ("cuda"if torch .cuda .is_available ()else "cpu")
    model .to (device )
    model .eval ()

    wrapper =textattack .models .wrappers .HuggingFaceModelWrapper (model ,tokenizer )

    def predict_forward (inputs ,attention_mask =None ):
        return model (inputs ,attention_mask =attention_mask ).logits 
    lig =LayerIntegratedGradients (predict_forward ,model .distilbert .embeddings .word_embeddings )

    nltk .download ('punkt',quiet =True )
    nltk .download ('punkt_tab',quiet =True )
    nltk .download ('averaged_perceptron_tagger_eng',quiet =True )

    return model ,tokenizer ,device ,wrapper ,lig ,predict_forward 

model_a ,tokenizer_a ,device ,wrapper_a ,lig ,predict_forward =load_ai_models ()




def run_textattack (recipe ,text ):
    dataset =textattack .datasets .Dataset ([(text ,1 )])
    attack_args =textattack .AttackArgs (num_examples =1 ,disable_stdout =True ,silent =True )
    attacker =textattack .Attacker (recipe ,dataset ,attack_args )
    results =attacker .attack_dataset ()
    return results [0 ].perturbed_result .attacked_text .text 

def attack_syntax (text ):
    tokens =word_tokenize (text )
    tags =nltk .pos_tag (tokens )
    new_tokens =[]
    for i ,(word ,tag )in enumerate (tags ):
        if tag .startswith ('JJ')and (i ==0 or tokens [i -1 ].lower ()not in ['not','no','never']):
            new_tokens .append ('not')
        new_tokens .append (word )
    perturbed_core =" ".join (new_tokens ).replace (" .",".").replace (" ,",",")
    prefix ="I don't think that "
    return prefix +perturbed_core [0 ].lower ()+perturbed_core [1 :]




def get_captum_html (text ):
    inputs =tokenizer_a (text ,return_tensors ="pt",truncation =True ,padding =True ).to (device )
    baseline =torch .full_like (inputs ["input_ids"],tokenizer_a .pad_token_id ).to (device )
    attributions =lig .attribute (inputs =inputs ["input_ids"],baselines =baseline ,additional_forward_args =(inputs ["attention_mask"],),target =1 )
    attributions =attributions .sum (dim =-1 ).squeeze (0 )
    attributions =attributions /torch .norm (attributions )

    with torch .no_grad ():
        probs =F .softmax (predict_forward (inputs ["input_ids"],inputs ["attention_mask"]),dim =-1 ).squeeze (0 )
        pred_class =torch .argmax (probs ).item ()

    vis_record =viz .VisualizationDataRecord (
    word_attributions =attributions .cpu ().detach ().numpy (),
    pred_prob =probs [pred_class ].item (),pred_class ="Positif"if pred_class ==1 else "Négatif",
    true_class ="N/A",attr_class ="Positif",attr_score =attributions .sum ().item (),
    raw_input_ids =tokenizer_a .convert_ids_to_tokens (inputs ["input_ids"][0 ]),convergence_score =0.0 
    )
    return viz .visualize_text ([vis_record ]).data 

def get_lime_html (text ):
    def predict_proba (texts ):
        if isinstance (texts ,(np .ndarray ,tuple )):texts =[str (t )for t in texts ]
        inputs =tokenizer_a (texts ,return_tensors ="pt",padding =True ,truncation =True ).to (device )
        with torch .no_grad ():
            return F .softmax (model_a (**inputs ).logits ,dim =-1 ).cpu ().numpy ()

    explainer =LimeTextExplainer (class_names =["Négatif","Positif"])
    exp =explainer .explain_instance (text ,predict_proba ,num_features =8 ,num_samples =200 )
    return exp .as_html ()

def get_shap_html (text ):
    def predict_proba (texts ):
        if isinstance (texts ,(np .ndarray ,tuple )):texts =[str (t )for t in texts ]
        inputs =tokenizer_a (texts ,return_tensors ="pt",padding =True ,truncation =True ).to (device )
        with torch .no_grad ():
            return F .softmax (model_a (**inputs ).logits ,dim =-1 ).cpu ().numpy ()

    explainer =shap .Explainer (predict_proba ,tokenizer_a ,output_names =["Négatif","Positif"])
    shap_values =explainer ([text ])
    return shap .plots .text (shap_values ,display =False )




st .sidebar .title ("Red-Team NLP Dashboard")
page =st .sidebar .radio ("Sélectionner la vue :",["Rapport de Diagnostic","Laboratoire Dynamique"])
st .sidebar .markdown ("---")
st .sidebar .markdown ("****")




if page =="Rapport de Diagnostic":
    st .title ("Cartographie Statique des Vulnérabilités de DistilBERT")
    st .markdown ("Restitution visuelle et analytique des expérimentations de la Phase 1.")




    st .header ("1. Diagnostic Niveau Lettre : Fautes Typographiques")

    col_l1 ,col_l2 ,col_l3 ,col_l4 =st .columns (4 )
    col_l1 .metric ("Taux de Réussite (ASR)","100 %","Critique",delta_color ="inverse")
    col_l2 .metric ("Moyenne de Requêtes","17.75","Très Rapide")
    col_l3 .metric ("Mots Perturbés","31.46 %","Furtif")
    col_l4 .metric ("Précision sous Attaque","0 %","-100 %",delta_color ="inverse")

    with st .expander ("Exemples de cassure du Subword Tokenizer (WordPiece)"):
        typo_df =pd .DataFrame ({
        "Phrase Originale":[
        "The acting was terrible and the plot was boring.",
        "A visually stunning film with a great soundtrack.",
        "The cast performed brilliantly in this cinematic masterpiece."
        ],
        "Phrase Perturbée (Typo)":[
        "The acting was errible and the lot was boirng.",
        "A visually stnuning film with a Mreat soundtrack.",
        "The cast pzerformed brilriantly in this cinematic mastrpiece."
        ],
        "Impact Modèle":[
        "Négatif 100% ➔ Positif 91%",
        "Positif 100% ➔ Négatif 100%",
        "Positif 100% ➔ Négatif 98%"
        ]
        })
        st .dataframe (typo_df ,use_container_width =True )




    st .header ("2. Diagnostic Niveau Mot : Substitutions de Synonymes")
    st .markdown ("""
    Cette section intègre la cartographie mathématique de la Grid Search effectuée sur les paramètres de préservation du sens global (`USE`) et de qualité des synonymes (`CosSim`).
    """)

    use_thresholds =['0.5','0.7','0.9']
    cossim_thresholds =['0.5','0.7','0.9']

    asr_data =np .array ([
    [80.0 ,80.0 ,0.0 ],
    [60.0 ,40.0 ,0.0 ],
    [0.0 ,0.0 ,0.0 ]
    ])

    conf_drop_data =np .array ([
    [61.3 ,63.4 ,0.0 ],
    [48.7 ,24.3 ,0.0 ],
    [0.0 ,0.0 ,0.0 ]
    ])

    fig ,(ax1 ,ax2 )=plt .subplots (1 ,2 ,figsize =(14 ,5 ))
    fig .patch .set_facecolor ('none')


    sns .heatmap (asr_data ,annot =True ,fmt =".1f",cmap ="YlOrRd",xticklabels =cossim_thresholds ,
    yticklabels =use_thresholds ,ax =ax1 ,cbar =True ,annot_kws ={"size":11 })
    ax1 .set_title ("Taux de Réussite de l'Attaque (ASR %)",color ='white',fontsize =12 ,pad =10 )
    ax1 .set_xlabel ("Qualité du Synonyme (Cos Sim)",color ='white')
    ax1 .set_ylabel ("Préservation du Sens Global (USE)",color ='white')
    ax1 .tick_params (colors ='white')


    sns .heatmap (conf_drop_data ,annot =True ,fmt =".1f",cmap ="Reds",xticklabels =cossim_thresholds ,
    yticklabels =use_thresholds ,ax =ax2 ,cbar =True ,annot_kws ={"size":11 })
    ax2 .set_title ("Chute de Confiance Moyenne (%)",color ='white',fontsize =12 ,pad =10 )
    ax2 .set_xlabel ("Qualité du Synonyme (Cos Sim)",color ='white')
    ax2 .set_ylabel ("Préservation du Sens Global (USE)",color ='white')
    ax2 .tick_params (colors ='white')

    plt .tight_layout ()
    st .pyplot (fig )

    st .markdown ("""
    <div class="custom-box box-insight">
        <strong>Analyse du Compromis :</strong> On observe visuellement la diagonale de robustesse. Le modèle ne fléchit que dans le coin supérieur gauche (filtres permissifs à 0.5). Dès qu'on exige une qualité stricte (0.9), l'attaque est totalement neutralisée (0%).
    </div>
    """,unsafe_allow_html =True )




    st .header ("3. Diagnostic Niveau Phrase : Négations Imbriquées")
    st .markdown ("""
    **Analyse statistique du lot de test** : À partir des résultats du générateur d'attaques syntaxiques automatiques, nous avons dressé le bilan de performance précis du modèle face aux paradoxes structurels.
    """)

    col_p1 ,col_p2 ,col_p3 ,col_p4 =st .columns (4 )
    with col_p1 :
        st .metric ("Phrases Globales Testées","5")
    with col_p2 :
        st .metric ("Attaques Réussies (Succès)","3","Sur Phrases Positives",delta_color ="inverse")
    with col_p3 :
        st .metric ("Attaques Échouées (Échecs)","2","Sur Phrases Négatives")
    with col_p4 :
        st .metric ("Taux de Réussite Global","60 %","Vulnérabilité Asymétrique",delta_color ="off")

    col_graph ,col_text =st .columns ([3 ,2 ])

    with col_graph :
        stats_df =pd .DataFrame ({
        "Classe d'Origine":["Phrases Positives","Phrases Négatives"],
        "Taux de Succès de l'Attaque (%)":[100.0 ,0.0 ]
        }).set_index ("Classe d'Origine")
        st .bar_chart (stats_df ,color ="#FF4B4B")

    with col_text :
        st .markdown ("### Journal de bord du Lot de Test")
        st .caption ("Détail des comportements asymétriques observés :")
        st .markdown ("""
        **Phrases Positives (Inversion Réussie à 100%) :**
        - *Wonderful/Touching story* ➔ Détecté comme <span class="text-neg">Négatif (100%)</span>
        - *Visually stunning/Great film* ➔ Détecté comme <span class="text-neg">Négatif (100%)</span>
        - *Cast performed brilliantly* ➔ Détecté comme <span class="text-neg">Négatif (100%)</span>
        
        <br>
        
        **Phrases Négatives (Résistance à 100%) :**
        - *Complete waste of time* ➔ Reste bloqué sur <span class="text-neg">Négatif (99%)</span>
        - *Acting was terrible* ➔ Reste bloqué sur <span class="text-neg">Négatif (100%)</span>
        """,unsafe_allow_html =True )




    st .header ("4. Spécification Théorique & Interprétation des Phénomènes")

    tab_l ,tab_m ,tab_p =st .tabs ([
    "Débriefing : Niveau Lettre",
    "Débriefing : Niveau Mot",
    "Débriefing : Niveau Phrase"
    ])

    with tab_l :
        st .markdown ("### L'effet de désintégration par le Tokenizer")
        st .markdown ("""
        Les graphiques **SHAP** ont mis en lumière le problème fondamental de la tokenisation **WordPiece** de DistilBERT. 
        Lorsqu'une faute de frappe humaine est introduite (ex: `mastrpiece`), le modèle ne la reconnaît pas globalement. Il la découpe en fragments isolés (`['mast', '##r', '##piece']`). 
        
        L'analyse de la valeur de Shapley prouve que ces fragments perdent instantanément leur poids sémantique positif et agissent comme du bruit thermique, provoquant l'effondrement de la prédiction à 0.019 (98% Négatif).
        """)

    with tab_m :
        st .markdown ("### Analyse de la robustesse sémantique")
        st .markdown ("""
        La comparaison des algorithmes au niveau mot révèle deux comportements distincts :
        * **TextFooler** triche sur la statistique globale en modifiant jusqu'à 75% du texte disponible pour forcer le basculement, créant des phrases absurdes pour un œil humain.
        * **PWWS** cible chirurgicalement l'importance locale (19% de modification), mais l'absence de filtres contextuels génère des structures aberrantes en remplaçant des adjectifs par de l'argot archaïque issu de WordNet (*bully*).
        
        **Conclusion :** Le modèle DistilBERT est immunisé contre les attaques sémantiques pures tant que les synonymes conservent une similarité cosinus supérieure ou égale à 0.9.
        """)

    with tab_p :
        st .markdown ("### Le Phénomène d'Asymétrie Attentionnelle")
        st .markdown ("""
        Pourquoi le générateur de négations imbriquées réussit-il à 100% sur les phrases positives mais échoue sur les phrases négatives ?
        
        * **Sur les phrases positives :** L'injection de la sous-structure `not stunning` ou `not great` focalise toute l'attention locale du transformeur. La carte **Captum** montre que le signal de négation locale écrase complètement la clause principale `I don't think that`, qui reste inactive (blanche).
        * **Sur les phrases négatives :** Des mots comme `waste` (gaspillage) ou `terrible` possèdent une charge statistique négative si massive dans la matrice de DistilBERT qu'elle agit comme une ancre. Même en encapsulant le mot dans un paradoxe syntaxique, le modèle reste magnétisé par le mot négatif brut. Le modèle se comporte alors comme un simple modèle de sac de mots (Bag of Words) déguisé.
        """)



elif page =="Laboratoire Dynamique":
    st .title ("Laboratoire de Simulation en Temps Réel")
    st .markdown ("Saisissez une phrase, lancez une attaque, et analysez la faille via nos outils d'Intelligence Artificielle Explicable (XAI).")

    user_input =st .text_input ("Phrase à analyser (en anglais) :","The cast performed brilliantly in this cinematic masterpiece.")


    col1 ,col2 ,col3 ,col4 ,col5 =st .columns (5 )

    if col1 .button ("1. Prédiction Originale"):
        st .session_state .original_text =user_input 
        st .session_state .current_text =user_input 

    if col2 .button ("2. Attaque TextFooler"):
        st .session_state .original_text =user_input 
        with st .spinner ("Génération de l'attaque TextFooler..."):
            recipe =textattack .attack_recipes .TextFoolerJin2019 .build (wrapper_a )
            st .session_state .current_text =run_textattack (recipe ,user_input )

    if col3 .button ("3. Attaque PWWS"):
        st .session_state .original_text =user_input 
        with st .spinner ("Génération de l'attaque PWWS..."):
            recipe =textattack .attack_recipes .PWWSRen2019 .build (wrapper_a )
            st .session_state .current_text =run_textattack (recipe ,user_input )

    if col4 .button ("4. Fautes de Frappe"):
        st .session_state .original_text =user_input 
        with st .spinner ("Génération des fautes de frappe..."):
            trans =textattack .transformations .CompositeTransformation ([textattack .transformations .WordSwapNeighboringCharacterSwap (),textattack .transformations .WordSwapRandomCharacterDeletion ()])
            attack =textattack .Attack (textattack .goal_functions .UntargetedClassification (wrapper_a ),[textattack .constraints .pre_transformation .MinWordLength (4 )],trans ,textattack .search_methods .GreedyWordSwapWIR ("unk"))
            st .session_state .current_text =run_textattack (attack ,user_input )

    if col5 .button ("5. Négations Imbriquées"):
        st .session_state .original_text =user_input 
        with st .spinner ("Génération de l'attaque syntaxique..."):
            st .session_state .current_text =attack_syntax (user_input )


    if st .session_state .current_text :
        st .markdown ("---")

        st .markdown ("### Résultat de l'Attaque")
        st .markdown (f"**Texte Original :** {st .session_state .original_text }")
        st .markdown (f"**Texte Actuel ➔** `{st .session_state .current_text }`")


        inputs =tokenizer_a (st .session_state .current_text ,return_tensors ="pt").to (device )
        with torch .no_grad ():
            probs =F .softmax (model_a (**inputs ).logits ,dim =-1 ).squeeze ().cpu ().numpy ()

        conf =probs [1 ]*100 
        pred_label ="Positif"if probs [1 ]>0.5 else "Négatif"
        pred_color ="text-pos"if pred_label =="Positif"else "text-neg"

        st .markdown (f"### Prédiction du Modèle : <span class='{pred_color }'>{pred_label } ({conf :.1f}%)</span>",unsafe_allow_html =True )

        st .markdown ("---")


        st .markdown ("### Explicabilité (XAI) de la décision")
        st .markdown ("Sélectionnez l'outil d'analyse à lancer pour éviter de surcharger la mémoire :")

        col_x1 ,col_x2 ,col_x3 =st .columns (3 )


        if col_x1 .button ("🔍 Analyser avec Captum"):
            with st .spinner ("Calcul des gradients internes en cours..."):
                captum_html =get_captum_html (st .session_state .current_text )
                st .markdown ("*(Approche White-Box)* : Cartographie l'importance locale. Vert = Positif, Rouge = Négatif.")
                components .html (captum_html ,height =150 ,scrolling =True )

        if col_x2 .button ("🔍 Analyser avec LIME"):
            with st .spinner ("Génération des perturbations globales LIME en cours..."):
                lime_html =get_lime_html (st .session_state .current_text )
                st .markdown ("*(Approche Black-Box)* : Détermine le poids statistique de chaque mot entier.")
                components .html (lime_html ,height =400 ,scrolling =True )

        if col_x3 .button ("🔍 Analyser avec SHAP"):
            with st .spinner ("Calcul des valeurs de Shapley en cours..."):
                shap_html =get_shap_html (st .session_state .current_text )
                st .markdown ("*(Théorie des Jeux)* : Observe la façon dont le Tokenizer fragmente les mots.")
                components .html (f"<div style='background-color: white; padding: 20px; border-radius: 6px;'>{shap_html }</div>",height =300 ,scrolling =True )