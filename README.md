# sexual-harassment-classification
sexual harassments
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from wordcloud import WordCloud
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from seaborn import heatmap, kdeplot
from progressbar import ProgressBar
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
from sklearn.model_selection import StratifiedShuffleSplit as SSS
import seaborn as sns
from sklearn.feature_selection import f_classif
from sklearn.model_selection import ParameterGrid
from IPython.core.interactiveshell import InteractiveShell
import warnings
warnings.filterwarnings('ignore')

InteractiveShell.ast_node_interactivity = "all"
from sklearn.metrics import log_loss
Read a dataset from csv
df_comment = pd.read_csv('./data/sexual_harassment/commenting.csv')
df_group = pd.read_csv('./data/sexual_harassment/groping.csv')
df_ogling = pd.read_csv('./data/sexual_harassment/ogling.csv')
df_comment.shape, df_group.shape, df_ogling.shape
((7201, 2), (7201, 2), (7201, 2))
df_comment.head()
Description	Category
0	Was walking along crowded street, holding mums...	0
1	This incident took place in the evening.I was ...	0
2	I WAS WAITING FOR THE BUS. A MAN CAME ON A BIK...	1
3	Incident happened inside the train	0
4	I witnessed an incident when a chain was bruta...	0
df_group.head()
Description	Category
0	Was walking along crowded street, holding mums...	1
1	This incident took place in the evening.I was ...	0
2	I WAS WAITING FOR THE BUS. A MAN CAME ON A BIK...	0
3	Incident happened inside the train	0
4	I witnessed an incident when a chain was bruta...	0
df_ogling.head()
Description	Category
0	Was walking along crowded street, holding mums...	0
1	This incident took place in the evening.I was ...	1
2	I WAS WAITING FOR THE BUS. A MAN CAME ON A BIK...	0
3	Incident happened inside the train	0
4	I witnessed an incident when a chain was bruta...	0
Creating Binary Classification Labels
we are concatenating all the data file in to one
z = pd.concat([df_comment,df_group,df_ogling],axis =1).sum(axis =1)
y = [0 if i == 0 else 1 for i in z]
df = df_ogling.drop('Category',axis =1).assign(y=y)
df_data = df_ogling.drop('Category',axis =1).assign(y=y)
df_data
Description	y
0	Was walking along crowded street, holding mums...	1
1	This incident took place in the evening.I was ...	1
2	I WAS WAITING FOR THE BUS. A MAN CAME ON A BIK...	1
3	Incident happened inside the train	0
4	I witnessed an incident when a chain was bruta...	0
...	...	...
7196	There was this person near a construction site...	1
7197	He threatened me by making inappropriate conve...	1
7198	happened during morning at university metro st...	1
7199	one day my aunt was returniec frm office .. sh...	0
7200	was victim of sxual assault RAPE	0
7201 rows × 2 columns

# check for null values in the dataset
df_data.isnull().any()   
Description    False
y              False
dtype: bool
### now let's look at the distribution of the two classes for train set
total = df.shape[0]
count = [sum(df.y == 1), sum(df.y == 0)]
label = ['1', '0']
fig = plt.figure(figsize = (10, 5))
plt.title(" Distribution of class labels")
plt.xlabel("----------Classes--------")
plt.ylabel("Count")
sns.barplot(label,count) 
plt.show()
Text(0.5, 1.0, ' Distribution of class labels')
Text(0.5, 0, '----------Classes--------')
Text(0, 0.5, 'Count')
<AxesSubplot:title={'center':' Distribution of class labels'}, xlabel='----------Classes--------', ylabel='Count'>

Observation :
Imbalanced data refers to those types of datasets where the target class has an uneven distribution of observations, i.e one class label has a very high number of observations and the other has a very low number of observations.

Feature Engineering
Text preprocessing
def fn_preprocess_text(sentence):
    stop_words1 = set(stopwords.words('english'))
    stop_words2 = set('and the was to in me my of at it when were by this\
    with that from there one for is we not so are then day had all'.split())
    stop_words = stop_words1 | stop_words2

    stemmer = PorterStemmer()
    text = str(sentence).lower()
    
    text = text.replace("won't", "will not").replace("cannot", "can not").replace("can't", "can not")
    text = text.replace("what's", "what is").replace("it's", "it is").replace("i'm", "i am")
    text = text.replace("he's", "he is").replace("she's", "she is").replace("'s", " own")
    text = text.replace("'ll", " will").replace("n't", " not").replace("'re", " are").replace("'ve", " have")
    text = text.replace("?", "").replace("i'm", " i am").replace("what's", " what is")
    
    text = re.sub('[^a-zA-Z0-9\n]', ' ', text) #------------------- Replace every special char with space
    text = re.sub('\s+', ' ', text).strip() #---------------------- Replace excess whitespaces
    
    text = text.split()
    text = [i for i in text if i.lower() not in stop_words]
    singles = [stemmer.stem(plural) for plural in text]
    single = ' '.join(singles)
    return single.lower()
txt = []
pbar = ProgressBar()
z = len(df_data)
for idx in pbar(range(z)):
    z = df_data.Description[idx]
    z = fn_preprocess_text(z)
    print(z)
    txt.append(z)

df1 = pd.DataFrame().assign(txt = txt, y = y)
df1.shape
  1% |                                                                        |
walk along crowd street hold mum hand elderli man grope butt turn look h7m look away 12 yr old
incid took place even metro two guy start stare
wait bu man came bike offer liftvto young girl
incid happen insid train
wit incid chain brutal snatch elderli ladi incid took place even
walk jorpati saw boy tri t0 take pictur girl n girl didnt know n itel abt repot polish nearbi jorpati
heard indec peopl make indec comment
enter subway dark decid go back suddenli man came amp hold hand amp forc go subway also ask number lajpat nagar subway
sexual invit
poor street light
catcal pass comment two ghastli thing delhi polic intern airport put friend appal protector law enforc airport make someon uncomfort
happen bu public transport broad daylight way colleg bu crowd enough space two peopl stand back back big bag em particular male stand next seat enough space behind noooooo start rub crotch shoulder subtli first obvious made discomfort appar seem get shift insid seat think ladi besid knew go say word neither peopl behind bu conductor well neither first place frozen grip shock fear think better prepar make shit go someon tri feat never know
night return hous parti saw boy tri touch young girl girl afraid cri came near boy run suddenli
peopl use stare inappropri way toler happen morn night
local colleg guy comment
make weird nois laugh girl
gone get ice cream notic 5 men look drunk turn around start walk away made sexual invit afar
comment stare go back p g afternoon hour
friend cross road
incid took place 4th septemb 2013 afternoon metro rohini east station two boy take pictur group girl sit opposit side
walk road two men bike pass lewd comment happen near vishvavidalaya metro station afternoon around 3pm
return home colleg boy comment whistl
sexual harras offic hour
return home workplac even drunk guy intent tri fall upon later saw polic nearbi got scare went away happen near rohini west metro station
harass
catcal comment st bu stand morn afternoon
walk home school walk boy came infront told love didnt knew guy ignor continu follow bodi help
guy misbehav us
teacher beat pupil
culprit stare group girl stare comment cheap way incid happen afternoon
shock incid report pink citi eve teas common happen girl almost everi well let share 1 incid friend studi 1 prestigi girl colleg jaipur stay pay guest cscheme jaipur two friend went market get someth suddenli 1 car came think 45 boy car tri drag friend car grope wrong place luckili friend save shout dread still cannt forget incid 1 case kind incid keep happen
report made dadar street market exact locat bigg sale maitreen shop particip feel safe dadar street market due mani instanc stalk
comment take pictur even
saw man forc take girl know help around help either happen night thursday market
boy comment shout
touch breast back
boy touch girl public bu
step mother chase girl hous night
head work notic someon follow decid chang rout that shoout say prostitut didnt golden vagina
ogl facial express gurgaon sohna
harass
biker mostli
enjoy ice cream famili good time suddenli felt someon grab hip first felt happen accident later go tortur twice scare death felt helpless scoundrel tall long hair dark complexion seem like drug addict recognis face still haunt
night chain phone snatch
group boy stand star us start move comment friend
sit desk work manag come ask updat stare chest femal also make indec remark newli join member regard sare
men see
impregn school boy use call everi time harass
guy stood front door obstruct way caus harass
tall boy almost regularli teas come gate catcal sing song indec manner
neighbour forc drug girl hous lock sexual assault
ogl touch gener compart dombivli train
group peopl teas girl mouth bad word
friend mine sexual harass bu though seat bu vacant old guy sat near keep touch intent want creat scene get bu though stop
man invit sexual pleasur happen
way school old man come toward grope breast hurt lot scare ran school told friend laugh made fun kept shut ever sinc
follow drunk man
man came close wait friend road said would give big money went sex
indec environ area lot drunker
saw boy go bike comment girl happen even
peopl comment take pictur daili busi
boy comment bad word
mumbai central station
walk around india gate guy constantli click pictur person went delet pictur happen afternoon
old person touch friend bu even lot rush
walk man came start tell nowaday stop talk told go hous
molest murga chowk 2 men snatch dupatta start touch inappropri broad daylight horrifi safe even
go buy kerosen man stop cover mouth tri touch breast
breast grope
person delhi metro take pictur girl cell phone
chase bike
ladi hostel area mani guy stand comment show intern part cross area plz some1 help make area free idiot touch walk bike pathet
walk even guy pass comment quot oye hoy kya lag rhi hai quot
afternoon travel metro lot touch gener compart
stare touch
incid took place bu two guy click pictur happen afternoon
guy follow pass comment
happen night
stare
follow comment man walk street
noth happen yet aunt teas group boy go colleg
whistl pass valgur comment
show vulgal pictur fb mg rel
verbal sexual harass sexual invit touch etc
touch inappropri metro afternoon
chain snatch man nearli 35yr old
colleg engli teacher use touch hand back stare badli dont know got number stare call everyday due switch phone sever month
boy snatch chain woman
come school met boy way call friend tell irrelev thing
man show privat place scooter act though noth happen
man touch bodi part play holi
men catcal comment im either way school way back home
conductor said bad comment insid bu
go back home guy star kept comment busesdhaula kuan
buddh bazaar shop man whistl walk side give wrong express
indec behaviour
public transport touch grope happen time stupid person present everywher delhi
group guy bike pass comment girl happen even
friend mine rape plan group boy date date boyfriend way never knew plan boy came took field rape went unconsci
incid happen 40 foota road vika nagar phase 3 2 men molest friend street near hous
tap boy came pour water found 14 year old girl wait fetch water grab hand drag away
car park particularli scari public space dark dusk pleasant would avoid
teacher use assault told mother ask go tuition
cheap comment whistl board metro morn
tuition teacher use touch chest use talk sex show pornograph video
ogl facial express comment
comment regular basi near club gym
guy wistl friend way
walk aunt place saw group boy call come refus walk touch breast want react start insult
metro pass cheap comment follow tri take pictur happen rajiv chowk metro station afternoon
go home time group boy bull take pictur girl boy girl time girl went home
physic touch privat part even time crowd street newroad
  2% |#                                                                       |
man felt friend hand genit
public vehicl boy touch bodi
call us local name like chinki
yester even wife com vastrapur nandan baugh way guy motorcycl catcal comment 9 pm sinc strech bopal444 applewood isol request author tri patrol road mani ladi go work come home late till 9 30
boy alway harrass use return home colleg
go home uncl pass hit front move ahead
girl go school sunday boy comment pass depress went home rather go school
mate two men way home school told feel like sex
stake indec messag friend north east
bu stop would board bu three car would pass sometim even follow school
lol happen everyday actual use
like eveteas noth great extent two guy tri follow way back home behav indec
pinch return colleg bu 29c two stop told exactli word english could slap hand full bag back also 29c also pretti crowd even enough space anyth wish stop
come class saw group boy teas girl
group girl us travel public vehicl conductor vehicl teas us
walk road bike came behind pull hand drive full road
touch dtc bu gagan vihar afternoon 24th august 2012
grant road station intent grope elbow man kept oggl till took local train
comment boy take pictur
guy whistlig comment
comment ogl catcal local market area quotwin beer shopsquot specif back area netaji subhash palac complex bylan etc
street harassmenta month ago come back buea bu station guy suddenli snacth travel box ask return box start insult stop gave want insult also
night wit than2 3 time
colleg friend comment short dress canteen said instig bad thing
three guy comment girl dress style
harass afternoon
two girl ogl chandivali junction
harass
guy whistl
roadsid boy hoot comment
saom women follow men night
passerbi pass leud comment even time
everi bridg crowd walk properli offic time morn daili men take advantag touch women singl polic personnel look happen continu happen sometim light work bridg difficult walk dark bridg
harass boy comment take pictur whistl
place sector 16 vashi navi mumbai 10 30 pm got sector 16 bu stop best 525 bu start walk toward home footpath shadi without street light decid walk along road walk someon loudli scream footpath quota idhar idhar dekhquot involuntarili turn direct see guy labour masturb smile shock wait around turn back walk away briskli spot light crowd kept follow sometim found head direct mani peopl turn around went way shaken reach home felt fear reflect feel punch good howev run away could think moment
friend suddenli guy start whistl went hurriedli happen even
girl call boy wink
way town realis neighbour follow kibera place work
stalk comment group men
walk toward hous guy start comment
stare man wait ola cab outsid gate 4 metro station stare linger made thoroughli uncomfort
78 peopl follow whistl ask number
travel bu person stand behind fulli touch bodi part point said bu crowd that happen could feel rush
comment rangpuri even
woman harass mutengen thiev dealer money taken
church amn start call sweeti babi great fear approach
public space saw man offens action
walk friend boy street shout say size
om sai beach hut safe femal solo travel owner gaurav bhagat drunk told staff watch see sleep friend stay seper room call check entir staff give inform activ whereabout bother hit eventu staff watch report make feel safe fact mani doubt overal safeti place hut bathroom hole wall someon could see insid overlook doubt someon watch slept chang cloth thing requir privaci recommend place anyon know told
three boy normal sit spot call quotbiasharaquot wait pass comment dress whistl ignor insult
poor light walk chan turn war catcal well
street light
incid took place septemb 2012 morn around 1011am girl drive twowheel two boy hit activa scooti softli touch back swept cloth
travel bu delhi merut highway nh 58 site somewher middl bu row posit second column 2 seater side girl age around 16 site next gap walk someth similar posit guy age 2425 walk passag stood us fill passag gap face toward girl back portion toward lower part face level girl approx level hi stand girl site push toward girl use bu movement mani time girl site face eye close terrifi left seat could give place person stand indirectli help girl walk littl distanc stood even tough seat vacant man took seat realli shame wish could done help girl
touch grope
travel bu teen age boy whistl well comment nonsens
travel bu girl sit near behind 23 boy stand pass cheap comment girl knowingli ignor need travel daili
even friend circl group guy stunt comment girl present circl
metro even hour guy tri touch
abus ask die
mother sick money buy medicin decid start wash cloth get money man ask wash cloth wash man came touch buttock
around 345pm bu 448 guy suddenli grab phone ran away
man touch girl bu
3 peopl includ sister reach destin decid take yellow line metro civil line boy sit see sister start whist sing cheap song
light
report friend mother wit rowdi biker make attempt snatch woman jewelri take routin walk woman rais alarm men came rescu tri chase biker sight right woman traumat resid decad nowher expect report matter polic
incid took place three month ago night rooster pub man late 40 indian cloth sat corner ogl girl pass lewd look comment minor
continu comment whistl outsid univers stadium north campu happen afternoon around 2 pm
misbehav
bu group boy start comment look pretti ignor start teas ask number
board matatu makina olymp yoiung ladi dress mini skirt tout togeth gy hang door start talk describ well made
catcal indec exposur
night around 8pm guy ask go place
fate even went visit friend mine began discuss togeth began go anoth level discuss start tell alway want friend began ask meant repli want girl friend immedi want sex moment refus insist thank god friend came hous save
man follow wednesday market go tri mislead figur home address spot market
passerbi touch us car even follow us
peopl scare chain snatch wit biker enter residenti area snatch chain purs wom en incid eve taes also increas day
eve teas common colleg comment teas etc
follow 5 boy teas follow even main street
girl wit somebodi click pictur although sure
go church sunday met drunkard old man call ignor start abus say quotmalaya wewequot mean prostitut
girl brutal beaten 23 boy unknown reason came forward help happen novemb 2012 even
eve teas use bad word
call terrac greed chocol follow bu stop 3 day
take advantag crowd guy tri touch even hour
walk back home suddenli group boy group around tri talk happen even
harass even gk
misbehav
found site recent guy kept walk behind quit time time look back would somehow disappear crowd step later felt someth jab hip spun around practic inch could get good look face due shaggi lank hair reach chin quickli back could rais alarm surpris part bu stand jam pack mani peopl must notic none bother interfer request pleas bewar filthi lout
physic touch man insid public bu
art craft group start send nude pictur girl start line prostitut
man tri sexual assault
market area crowd stop go men boy pass us touch us make excus dont enough place go also comment us
harass bu
34 men stare start comment sort eve teas
catcal comment touch other old delhi
whenev pass area shopkeep comment sometim give sexual invit
biker tri touch breast push fell bike hurt start abus call polic
teacher beat student
festiv newar saw live harass girl group boy touch part girl
boy lie took friend slept 3 day mum complain threw hous night
  4% |##                                                                      |
boy comment tri grope crowd place
best friend sexual harass neighbor six year old still live next inform anyon
way home school men sit idl whistl refus respond start abus
saw scene mall afternoon girl stand outsid mcd two boy came start comment repli start call actress name start whistl
friend molest neighbour inform parent immedi taken task
come back flat job greater noida two boy follow auto say quothow much charg night quot
realli bad
boy grope stare like hell
men gang stare even pass comment make difficult us use block e main road lead home
whenev friend pass boy sit around start comment sing song etc
left friend place went meet friend ladi welcom nice continu chat put ponograph video play reali made feel uncomfort sinc friend friend
two guy follow friend metro board got stare us throughout
stalk man market street live tri mislead
father brother gopi krishna movi hall ticket counter bit crowdi guy touch girl girl immedi repli shout later guy felt shame backoff
comment ogl catcal someth usual happen metro
friend rape tuition teacher
local chawl men call name caus seper husband
sexual harass morn go school friend guy appraoch us start ask name friend unfortun friend spat face come back school guy attack us friend took us dark environ gave us beat ourliv collaps woke realis hospit
survey carri safec red dot foundat along safeti audit street market mumbai
young girl harass father hous without parent know later discov night man tri kiss girl know girl report parent think inlov man
take advantag crowd rush hour
exboyfriend call blackmail girl
case classmat read school sunday morn unknown man appear widow class man stood long time watch girl read start ask girl question enter class cutlass ask girl undress girl also play trick escap wound hand man
famili near amus park class 9th know much sexual harass person walk past touch inappropri
feel safe
girl grope peni
articl written last year situat took place bare enter 11th grade leav high school flashback time bodi empti canva boy splatter paint tell stori 10th grader gurgaon quotth stranger lip graze upon end cheek bruis arm pin dusti grey ground room kept alert eye outsid face unknown men uniform smirk stranger slid snakelik finger bodi struggl undo button could feel eye turn red bit hard shoulder small voic said leav dont like felt empti liquor bottl leg hand flesh jerk push asid head buzz alcohol panic manag stand look around realli want straighten press frail bodi upon nake wall push asid ran toward open hall men age stood rail oblivi act cheek burn forc kiss shame hold shoe hand scurri staircas search known face look run reach cold lobbi offic stare entranc strang glare anxiou face shove stiff self close elev seek unfasten pair short settl forc slither finger waist whisper go finger like asshol whimper hop away felt back cloth seek phone wasnt rush panic exhaust fled offic hallway rummag around build explor everi step took earlier search prize possess eye burn even tear blind vision settl raw cheek behind rush downstair head busi road broad daylight scuttl barefoot search friend late gone stranger behind grab shirt strong hand rip cloth bare back becam expos blink light stare million nameless men cri help nobodi care enough hear ran faster without possess phone money bag gaze crowd nearbi came rest staircas tear ran face unknown figur settl around head knew whisper strive punch familiar number phone rememb moment child gone astray parent cram market place moment devoid thought thought worth get back safe rememb worri face moment magnitud shook back forth glanc abund face unabl see delic featur muddl concern men women walk past put forward help hand trust matter minut agon face rush past stood ran toward open arm world stood still held took accustom smell cologn felt homesick wave crash wall head im scare he go find said enough know took grey shirt bag walk eyesight alien peopl wear murmur prop hand shoulder pull close rescu friend clutch stranger friend depart curs locat ride end hous held face low voic inaud friend next room ask favour dont anyth stupid said anxious read mind state friend beyond categori wast follow given duti pick marbl floor led toward societi gate wave brief goodby timidli walk away sometim dont need word explain feel sometim know like knew enfold scent shirt felt possibl petrifi episod face made easier breath sometim dont need date love somebodi month feel comfort dont need kiss even around time nervou presenc linger around moment enough demonstr establish brief connect safe even minut sorri glad said word phone formal stand park watch walk toward gate could feel happen may episod import chang everyth im sorri danc end arm swear world stood imposs still quoti attempt suicid hour later save live strong tell stori
man regular bu rout got stop touch wrong place felt bad
pinch breast water park mehsana gujarat
four boy whistl comment girl happen night
walk street guy stand lean gate home whistl call whole time cross home take road daili except sunday go music class went month final quit music class 13
meet meet went asid talk phone certain man came start urin fenc anoy
age 19 yearstyp harassmentstalk leh market time caltcal choglamsar near choglamsar bridg even
father neighbourhood engag sexual activ daughter forc
uncl beat wife street outsid hous help ladi later went insid assault lock
night 2 3guy comment cloth
boy stare catcal go home school
sever girl rescu video den mashimoni villag exploit pornograph film video hall
travel bu man stand behind tri touch
poor street light
amazingli rude face given buy guy afternoon hour
friend tri kiss forcibl park
report regard survey conduct sexual harass differ form particular area
somebodi tire touch sensit part
boy area make indec pose follow girl scooti speed bike make sound
stick touch bu crowd whistl walk home
attack tri grab gold chain push
group worker click pictur run park multipl pictur taken call local secur guard snatch phone delet pictur
man tri touch woman
gener coach metro kashmer gate saw group boy pass comment girl happen even around 5
around 8 pm got vishwavidyalaya metro station get back hostel saw guy follow minut pull back ask friend touch inappropri shout ran without look back hostel
middl age man comment dress also tri touch way home school
travel metro friend afternoon two person came start comment
walk mall friend middl age man ride bike watch us make inappropri face pass comment us annoy youngster grownup also respect women
girl touch aimlessli boy school never report teacher
incid took place even outsid mall boy pass lewd comment girl stand alon
teacher hit student chick
drunk men harass
boy next home alway stare
sexual harass friend attend even school way home took bike park reach hous bike man carri rape plantat touch unabl get found morn taken hospit
saw two boy bike come opposit direct snatch golden chain middl age woman cri loud get back vanish second place
harass metro feeder
incid took place 20th septemb around 530 pm go class got car guy push expos privat part bash along polic
woman thrown hous babi without food two week depend passersbi babi feed harass take advantag hoodlum child expos danger
survey carri safec red dot foundat along safec audit street market mumbai
person snatch chain
chain snatch due por street light
sexual abus friend come back villag farm man caught abus sexual
25 th dec pass group boy call ignor persist follow ignor sudden start call name
happen minut street two guy came bike snatch ladi chain ran away shock happen bright noon public around
friend molest new year eve
stay hostel near incom tax often guy pass lee comment stare girl way make uncomfort mani time
cab driver kept brush leg kept ask number
play hide seek game hide neighbour hous peep bed sex anoth kid report nobodi believ said liar
sing
harass
acid attack
use face almost everyday internship period
pass street guy whistl
incid took place 15th septemb 2013 around 7am happen park near kanhaiya nagar metro station mother teas unknown person
class 12 return tuition class boy pass lewd comment
uncl harass alon
return home market boy stand near pass vulgar comment
incid happen wednesday market man constantli stare follow around told vendor nearbi confront man went away happen even
post sunset get dark scar men loiter around feel unsaf
friend mine move along road 17th januari 2012 stroll along road man approach stop listen man told interest man got annoy pull near push bodi rescu rape girl ran away
number boy wistl comment walk street bhaktpur
vehicl go town kariobangi man seat next start touch breast felt indec alight bu
come back jamia millia wait auto stand group colleg boy pass filthi comment ignor min 2 boy approach said bad word made feel like cri live alon jamia hostel safe girl miss parent live bihar studi jamia sacr sometim cri class difficult explain feel even univ student pass filthi comment disciplin r either classmat see class colleg chang total illitarar
old man came near touch privat bodi part insid tempoo
happen even
black zen estilo tint glass follow throughout way back
  5% |###                                                                     |
construct worker along railway line anytim ladi pass nearbi normal comment catcal whistl face sever time
math teacher tri forc tri molest refus get person tuition
touch grope market
group young men sit roadsid everi even 5 p late even
saw guy take pictur losar new year parti villag
guy comment girl
someon felt back dadar station climb staircas stop shout slap
rape park
heard incid know true heard old ladi got rape young teen boy
shopkeep make diffrent face
man came near said breast nice want touch
girl rent flate n live alon thre taht hous owner rape cri n went polish report happend infort hous
school friend comment wrong
crowd street man grope chest twice attempt third time punch stomach ran away
use experi type categori younger veteran citi field avoid issu ogl facial express comment remain unchang year everi place travel local experi categori assault except rape sexual assault avoid rape year simpli never situat could lead learn live like prey anim world full predat
14 year girl physic harras boy touch breast
friend want take man hous end relationship
friend mine guwahati walk hostel main road via 6th 7th street gopalapuram guy motor bike tri pull bike hold scarf wear luckili let go scarf save
woman revolt man take advantag morn rush hour grope
man stalk caught start touch
went work 2 girl friend group boy 25yr tri divert attent toward ignor went back home luckili follow us
sarjapura road signal bu stop bangalor path bridg unsaf place past year 2012 2011 8 night mani hijda stand dress suspect sex worker path bridg poorli lit lack bright light bridg give cover complet unsaf women walk path night
happen 9th decemb http epap timesofindia com repositori ml asprefve9jufuvmjaxmi8xmi8xmsnbcjawmza0
walk street man comment
sunday afternoon church way back sibl huge man tri pull gown see quotinsidequot tri fight back seem strong drunk told say word would slap hard
whikl walk skywalk
friend visit salinadi sankhu two boy follow us start whistl
night ifac everywher delhi spell
walk tution alon guy start follow comment
slap unknown person went check metr came slap make believ open meter noth els mind pain tear hurriedli went home report mum
walk friend two guy start comment follow
uncomfort stare walk
guy 2530yr grab back gener compart metro get
harrass pass comment board train
guy tri touch privat part even
train vellor bhopal
comment
touch indec even morn
friend walk toward shop guy sit car open door car shout indec thing happen even
man comment tri call use name
travel crowd bu stand group boy enter bu boy stand behind tri touch back
14 year old acuint famili adult marri person would hold touch breast would know react would alway tri avoid never mention parent anyon shi express similar incid happen 12 year old two men famili acquint would act visit famili
pass lewd comment class
comment bad word even district center
girl come school head home call sweet man went rape man man call ran way
wa shop boy stand outsid shop call onto ladi pass came kiss forc
deboard metro rajiv chowk young guy rub twice initi ignor think due overcrowd metro kept repeat could feel bad touch turn back hit hard also give warn never repeat common incid rajiv chowk mani friend also prey
misbehaviour guy
girl rape uncl compplain mom tell stay shut endur lost digniti anyway
guy comment make facial express us walk cp happen even night
even make whistl road also follow
comment group boy rough word
go friend birthday wear dress 2 men pass dirti comment cloth
finish studi boy ask kiss street refus want intimid ran away
even 5 45 pm
whistl boy return home tutori class late even
go church certain man came ask name refus start follow
walk home alon group girl teas
lost count number time someon inappropri touch citi centr mani time look perplex react grate
went friend got late man came bought us drink thing rememb woke uthiru pant rip
guy show peni pass realli offens comment
neighborhood boy use touch small girl appropri caught punish
misbehav
man around 0 tri get hold back roadway bu travel back hometown
taxi driver stand outsid jalvayu tower apart flash 10 00 confront hit ran away doesnt make feel better
stalk call home visit
return home aunti hous boy micro bu comment touch
happen station wait train man start touch stare us women
frequent travel mumbai local train like mumbai women take harbour line train goe mahim cst seen mani time 8 pm train get empti presenc rpf personnel railway look increas secur rout
murder boyfriend want break
shop friend boy start comment us
man tri come close
touch metro
walk crowd patan found someon touch bodi crowd feast festiv go
guy purpos touch ran away
physic abus public vehicl st night time bu pack
man sit next bu tri take pictur phone
father friend uncl use stare use make uncomfort
friend pass mashimoni area men normal sit kiosk near railway chew mirror cale whistl
regular even walk gang start follow pass inappropri comment start take pictur
man known disturb girl sent market mother around 1pm met man way forc stand refus touch breast
survey carri safec red dot foundat along safeti audit street market mumbai
pass group boy stand side road suspect shout quothey beautyquot
survey carri safec red dot foundat along safeti audit street market mumbai
guy actual came face said quotter ko kacha chuba jaungaquotsick made numb
whistl area indira nagar
travel local bu saw guy pass neg comment young ladi alon feel embarrass
boy girl silent road boy comment girl enjoy silent road wherea girl scare walk fast
push crowd comment group happen even
guy touch inappropri place
gone wednesday market afternoon way back man follow back home got scare ran inform brother brother beat made apolog
report terribl incid happen girl know friend girl stay friend place gone meet boyfriend boyfriend trap invit friend took turn rape near toilet complex sanjay camp girl
market area went collect medicin guy bike snatch wallet could anyth
harass commun indira nagar
man make us girl game club go play mess alittl beat buttock start touch breast
travel street polic stare girl gossip nonsens girl
street light road dimli lit felt someon follow check right
area never safe girl night ladi go even back hous face unnatur stare men disturb
know girl studi class 10 stare boy stare would eat
know happen girl women face everyday men walk toward bump basic tri kinda physic touch say anyth blame place crowd never meant anyth obvious girl come know someth done purpos happen mistak hurri stop everi 5 step slap man tri ye frequent happen everi 5 step crash man walk toward
get back school afternoon boy gang comment confront got abus argument slap boy stun apolog behavior
rel mine touch indec manner time back cours disguish cute hug elder give young one left harass
go toward new delhi railway metro station guy ask go hotel
ogl whistl
ladi call bring cigarret man privat car brought cigarett man give money cling ladi hand talk low tone ladi realiz man hidden intens remov hand leav moneh man drop money drove away faster
comment etc common even
touch stare sexual invit comment
man grope public pretext bump
friend go road guy car pass make comment happen afternoon
friend enter lift old man alreadi insid old man tri kiss friend
near grm school
everyday makind street sexual harass sexual invit comment catcal occur littl children also victim heard rape issu spot reason lack street light harass hide dark commit evil
time felt uncomfort trvell public vehicl touch
station unsaf light dim sign badli damag women access emerg escap rout lot harass happen due presenc loiter stare women pass lewd remark
  7% |#####                                                                   |
boy group start stare whistl
call ask phone number name
friend went parti given alcohol friend got drunk guy took advantag helpless state rape could help later got pregnant due
classroom somebodi use phone sent porn video phone even left note watch someon love
man tri hug girl behind
colleg go shop felt miser unsaf stranger stalk also took pictur incid lyk must stop polic strict rule
sexual harass happen friend mine visit guy ask guy invit hous enter guy close door put music high volum forc bed rape
saw man touch girl privat part insid bu
incid took place even deep cinema market ashok vihar girl come institut attend class boy stand exit start whistl also ask number
happen afternoon even metro
lot auto driver metro station often misbehav talk rude even abus verbal feel unsaf sinc travel alon often due work colleg get late
old man shop comment us buy thing shop
get amut metro station found 2 boy stand outsid metro station ogl pass comment
stare bad intens pass street even face comment teas boy make feel uncomfort
friend random stranger comment someth bad
friend brother use call time 1 2 day propos refus start threatn like say ye face consequ like uthwalunga goli maar dunga lot
head home met boy wink abus also abus
even dtc bu 543 mobil stolen somebodi
happen friend afternoon stand outsid home man stare come toward came realli close touch grope around 6 month
travel bu cundoct tri touch bodi part sever time
pass road saw man near bridg rape 5year old girl
physic abus travel crow public vehicl return form offic
stood bu stop man step toe ask look hash eye decid say anyth know intens
two biker tri snatch dupatta near bu stop took dupatta away ran corner
touch wrong place train
stalk comment bike even
metro person continu stare avoid got uttam nagar station stood front door door close gave fli kiss show cheap gestur
saw man beat wife amp shout
realli bad
incid took place juli 2013 even two biker tri snatch chain woman unsuccess
walk sister old man tri catch hand got scare walk peopl
happen take girlfriend koti famou shop place hyderabad citi visit candi shop near big bazar guy star comment indec gf wear tshirt comment indec believ unsaf place india naresh
walk road person start flirt fell unsecur alway stay flirt way home
realli bad
get share auto destin men made kiss nois
friend got grope behind travel cst station
saturday go shop heard someon whistl turn see saw anoth boy come toward pretend seen walk away
happen near karol bagh metro station go class someon comment
drag van somehow manag escap help peopl happen even 8 pm
came colleg cab come attend 8 clock lectur colleg taxi driver told bug stuck hair tri brush bug told still told take shove hand insid tshirt quit frighten shove hand away got cab quickli possibl happen outsid gate sophia colleg mumbai
walk station someon touch
misbehav
harass near sukh sagar
purs snatch
men walk group saw begun touch privat part ignor hurriedli walk pass lough said quotstop pretend need thisquot
friend shop group og middl age men start comment us
went morn walk guy bike pass made obscen gestur repuls upset wonder provok gestur
come station exit station group boy pass comment stare contin
vulgur messag facebook
pass school boy start call good reason decid ignor
walk school men made facial express
comment ogl afternoon local dtc buse overcrowd men around pass cheap comment make uncomfort women travel dtc buse women separ women coach metro
go home guy comment dress n felt bad
sexual harrass
catcal comment touch
saw boy forc girl kiss also touch
group 45 boy pass bad comment girl wait auto incid took place two moth back even
boy click pictur
obscen comment
harass gang guy rape move street met group boy came confront tri escap ran rape
girl small incid happen stay parent orthodox famili guy also stay flat small room terrac notouri guy girl innoc everi took advantag guy use call often whenev seen terrac outsid hous use touch privat part also ask touch refus cruel girl scare tell incid anybodi esp parent fear put ito four wall sooner attain puberti came end atroci act girl allow go anywher alon guy caught land owner start continu viciou act beaten death member flat room lock perman
saw boy harass girl road even
drink near park peopl came tri rob us show us knife made nuisanc could get
saw girl harass group boy stare dress start whistl teas girl pass girl ignor follow met friend
man roam around powai stop women street first ask direct money claim dire need variou reason mani girl encount alarm
travel nepal yatayat stand middl age man tri touch back privat part could react time incid alway feel scare travel anywher public transport
misbehav girl go buy cloth
walk look butock say quottumetosha mbogaquot
sexual invit
friend go home boy tri ask lift
go hous dark alreadi guy follow look like gambler start comment scare time suddenli man known shout name went
harass
man flash broad daylight horrif
friend mine abus asa child
guy show peni street boudha make embarrass walk work
coupl singl women avoid spot 7 pm 10 reasonsther mani voyeur men come group sit rock close coupl seem video surveil entri road mean perpetr could get away koliwada villag stretch avoid 7 pm versova beach till main juhu beach
sit rickshaw guy came snatch mobil gave abus
peop call chinki abus north eastern horrifi
comment colleg afternoon hour
follow
pedicab cyclist insult threaten shopper outsid mall yell top macho voic
phone stalk year guy recollect interact work restaur societi got number work kept call block call continu send messag well year even found facebook
go buy thing cycl 2 men bike kept follow cycl sometim crash cycl shout thing ran away
incid took place may 2013 pervert comment inaptli woman metro station snap back crowd gather guy got beaten
girl beaten brother found go school use money given fare walk around bad girl
happen metro station afternoon
coupl biker went pass lewd comment make vulgar gestur
stalk classmat also polit parti member
boy came touch bodi ran away pain
walk toward station around 830 guy bike drove real close touch hand express also cheap
near south ex guy whistl constantli comment
incid took place even 7th may 2013 friend mine came club guy start comment dress use vulgar languag even follow
men like call us tell us want boob big
walk street two boy sit insid cafe call look kept go boy whistl tri follow walk fast lost sight
shopkeep start touch girl privat part window shop laptop bag
group four boy car outsid ramja colleg comment happen morn
dim light churchgat station plu dangl wire make danger
stalk borivali
incid attempt rape report day back
bu man purpos fell touch unppropri excus
around 40 year old man touch bodi travel microbu bu routegausala maharajgunj
chain snatch amp indec exposur
go school micro bu public vehicl boy tri touch back move bit far
  8% |#####                                                                   |
reach bu stop work head toward home biker pillion rider came along behind tri snatch chain abl deni somehow alon back
walk home school man touch butt
man age father attempt molest park lot littl kid
friendli man invit berth start touch inappropri
touch amp harass verbal
bunch loser weird guy kept tail friend femal shop lulu mall kochi kept follow us floor anoth even enter store went point even tri photobomb us got rid leav mall
colleagu use pass comment well ask go hotel
friend physic abus uncl
comment upon even hour
guy travel rickshaw hhe kept look back wierdli repeatedli
touch breast
field play certain boy start take pictur us
came back school reach home go upstair somebodi age 1823 ask whether compani deliv post build said start climb said innerwar shown act act adjust grope screamt ran away lost braveri confid let fight teach boy child manner let stop india
bu conductor pass bad comment
despit uniform call men sit along road chew khat
realli bad
walk road man comment movement girl bodi part indec languag quot
stranger touch butt
return friend hous follow stranger
sit bu guy next seat show peni oppen pant
indec song
chain snatch earli morn ladi morn walk
pick pocket snatch
accent number plate follow almost daili street poor light
usual face problem like teas comment sing song way home wait friend middl age man come near want talk ignor
friend went visit pashupatinath templ peer group came close us whistl comment us
aunt sent around 500 pm shop found besid hous buy candl sinc light arriv store mani peopl especi boy went anoth store bought candl way back saw figit without know actual talk held tri kiss scream
happen cousin said man tri touch said would buy scooti spent time
sexual invit stalk
metro station two boy daili follow pass comment
come back colleg guy follow bike follow till hous
night 2 3 boy tri touch friend
age 16 yearstyp harassmentstar bu 27th march 2015 time even catcal villag time
girl walk along railway line shout boy stand side railway quothey know work themquot
boy metro gener take pictur girl travel also give indec action express case recent happen travel metro
mcd rohini 34 boy sit front seat comment take pictur ad whistl took place even
worker slum nearbi region sexual assault teenag take advantag mental ill state
seen two boy comment friend teh comment cheap
unsaf area near metro station comment
ogl facial express especi smoke stare back aggress
happen gener coach metro even
two girl park follow person irrit pass comment touch unnecessarili
night auto driver took auto wrong direct fellow passeng start shout help get mess
touch inappropri way
poor street light
help bu help everyon get ticket hand tri finger bent finger backward point almost broke way bu pinch ass though fail shame make issu
boy car follow friend happen even
friday go home school tire hungri beauti ladi saw request help tri make ladi know tire hungri thought ignor start call name almost slap didnt insult get point even fight highli emabarass harass
ride scooti men bike touch hit back part
visit friend pass men stand outsid certain movi den begun stare us pass start catcal whisl friend laugh walk fast live friend behind man said friend key hous go fast anoth said look good bed
walk along roadsid man want touch burst stop next time met made sure touch butt annoy
survey carri safec red dot foundat along safec audit street market mumbai
incid took place month ago even around 8pm woman go two boy bike came snatch chain
man touch knee bad manner
bu stop stand line wait bu guy tri come close touch inappropri place
normal stare girl common
girl stand balconi face horrend sight man sit car show privat part ask join
guy continu click pictur metro even
agirl follow guy n say ju wan talk dont know that ran away complain parent
railway station unsaf
cross road friend group biker comment whistl
harass repeat indec insult messag
comment touch pictur taken dtc bu red ac bu even
sexual harass neighbour year back
went select citywalk wherebi leav boy follow us car comment us
guy take pic femal friend
bu
call pastor offic servic pastor tri touch privat part inform parent pastor arrest
alway tell walk girl beauss done arbot fell bad becaus accuss noth done
place crowd guy opportun touch girl uncomfort
chain snatch afternoon unknown man wednesday market immedi fled scare tell anyon
door door salesman vodafon come home around ask inform variou scheme offer alreadi number start call daili offici reason engag convers ask whether boyfriend etc alreadi told mother told call home pretext get scheme activ repli quotwhat get give packagequot told whatev ask kiss came home mom grand mom cousin boyfriend hide room soon came home start pound took polic station 9th std approx
lewd comment sexual invit
walk road saw man comment bad thing girl talk rough word girl
phone snatch market man tri push fell anoth man pick phone ran away
even around 4pm guy use indec languag
let teas girl goe wrong
malviya nagar market boy teas friend
friend mine live besid hous usaulli harass go read man name john alway harass advantag like idea
guy comment group girl comment ridicul
metro
someon came scooti behind victim touch intent
villag man age 25 rape 15 year old girl
saw 23 men snatch old woman chain
friend came back home night guy outsid bar panvel comment whistl
man teas young ladi road
even guy whistl cross cycl abus
near indra vihar hostel guy give weird look pass bad comment girl pass happen even
girl wait bu go colleg felt unsaf old man ogl made uncomfort
catcal eve teas near railway station jogeshwari grant road station happen mostli everyday travel
catcal comment ogl facial express
guy colleg continu click pictur even hour
call name padum zanskar
accus stalk victim return tution
touch
way school mate certain man told quotmi wife come kiss love
stalk
follow group boy movi hall auto stand
auto afternoon guy start comment badli
last month go back home school primari school girl rape old man
went fetch water stream man came touch buttock
realli bad
two men catcal outsid metro station
saturday last year 2014 aunt went fetch water around 500 clock morn around buea central market saw man stand besid farm bag man call come help put bag head unfortun know man intent beaten rape helpless could walk extend want die rush hospit man caught
five young boy fight street residenti disturb
stand stage town guy start insist board matatu like refus
comment even vishvavidalaya metro station
jaipur shop mom felt someon touch backsid turn around slap
 10% |#######                                                                 |
harass
slap man road said dress vulgar manner
two boy comment girl
ogl metro
live jp road near chai coffe previous known barista whenev walk past neighborhood alway see gang peopl stand outsid car park blare music make comment niec stand outsid wait someon boy came got scare also pan dabba next chai coffe see guy sit wall drink public polic know everyth noth scare complain live area
friend went department store rajasthan carri 2 bag good bought 2 men street told quothum utha le aap kano toyquot happen kishangarh near power hous
man kept fall metro elbow even brush breast felt intent
boss alway ask go outsid use take cafe restaur speak nonsens took dulikhel say pay salari
happen student time look dirti manner
happen crowd train mainli bridg
beauti ladi 2426 year harass group boy whistl comment stare n take pictur
go attend durga pooja guy tri touch hand
night group boy conmment
road trip man took pictur vehicl halt
survey carri safec red dot foundat along safeti audit street market mumbai
catcal whistl comment ogl facial express take pictur touch grope sexual invit
age 16 yearstyp harassmentfaci express neighbour hous even stalk leh market time catcal leh market villag timeeven
south ex friend guy start comment
incid took place movi hall afternoon guy abus girlfriend physic report cinema author strict action taken
friend park guy come friend call name ask want friend got scare
walk even guy car group start shout whistl
cross crowd guy came hit badli comment unbear
work team leader spoke derogatori comment friend absenc
follow group men
teacher beat pupil
walk rickshaw puller whistel comment
street home sever peopl light condit horribl bare street light littl amount light found come shop light even road good get flood littl drizzl
peopl take advantag overcrowd metro buse pass lewd comment grope
boy like brother almost 24 year age heinvit place strip nake told feel home uncomfort
gone home frm ktm baglung bu thre 4045 yr old guy next seat turn point road guy tri touch n bodi part two three time screeem abt n pax support n everi forc leav bu
group boy pass lewd remark us access street time
man held wrist near balaji chowk ask sexual favor push ran away
insid tempoo man touch bodi know
happen dwarka mod even guy motorcycl pass whistl kept stare minut
went govern offic offici appoint offici show pornograph video scare alon
harass
harras morn
light feel unsaf
someon follow tri walk side touch also tri rob chain snatch
went staffroom teacher apart mr ondigo told go hous scare ran away
two friend go colleg boy took photo didnt like
grope delhi metro
stand hold back seat guy know place hand mine
harass
chain snatch amp lewd behaviour
walk take pictur red fort group boy whistl happen even
friend lift man touch us
man pat friend walk ran away
happen friend mine went market buy veget way back met boy came bush start insult refus request
man masturb street next cigarett shop
harass
light drunkard
man tri touch woman bodi pack bu
ogl grope
friend go templ man start follow us sing cheap hindi song happen even
travel dtc bu even guy continu tri touch even stop rebuk
group boy start comment figur
catcal whistl
commut kurla kharghar daili pervert get train mankhurd govandi pass lewd comment everi girl station differenti teenag girl woman late 40 comment alik quit disturb fellow commut say noth get kurla start take chanc get railway bridg alway tri grope woman pretex crowd male passeng chose ignor even look shameless act strongli feel separ bridg station women
uncl own shop mom send buy stuff tri touch inappropri asham
group man teas girl way work
two men bike snatch chain woman wait husband come
mani youth drink open street abus girl peopl gather save girl child stop activ
girl use get rag everyon found ugli made depress peopl use harrass make fun know like
biker tri grope
father tri rape asleep
boy stare girl skirt zip open
villag girl rape first murder nobodi anyth convict bodi cut separ
stuck crowd thursday market low cast cheap touch inappropri 3 time breast ass abl find first crowd found shout nobodi came help pretend heard noth sadli ran away could slap
go home boy teas us comment us
peopl comment stare continu
cycl earli morn man came behind touch inappropri
harass girl ask date refus report friend even brother underlook happen twice last year last two year 2013 2014
man follow whistl walk fast lost crowd
infront home shop group boy make junction use target could noth brother heard activ scold stop
harass
peopl pass comment
walk colleagu man pull back wink mad walk away angri
realli bad
happen even near select citi walk mall
way home guy stare 6 month
travel bu near naraina ring road guy continu star amp also give bad facial express
go parti boy rout comment vulgar
group boy stand outsid colleg ogl uscom catcal
intent inappropriatelli touch walk
wit incid indec touch done guy happen even hour
wait bu guy came stood besid walk could get auto guy follow make lewd gestur panic stricken abl make mind whether call help take pepper spray fortun auto arriv moment save happen night
person contious stare tri touch bodi part insid public bu
man snatch chain rel tore cloth
catcal
ye sexual harass happen studi 2 nist colleg awkward moment life till
guy walk pass us walk girlfriend heard click phone know confront turn taken pictur made delet
would travel 271 bu often experienc men tri brush stare worst thing two three conductor alway tri touch women purpos hold hand women hand ticket happen four five time
girl school alway touch breast classmat never report
incid happen friend well walk road mask men bike
survey carri safec red dot foundat along safeti audit street market mumbai
girl sexual harass fellow classmat
guy misbehav
boy comment go templ
girl whistl
friend ogl man long time made weird facial express look
stand bu stop two guy bu stop crowd look around chang posit stood femal stand bu stop see face could recogn guy could cheap thing stand near ladi wait bu guy also came near toa like also chang posit like notic action saw give cheap facial express see chang direct face opposit direct also notic contin smile look talk tri judg board bu also board bu boy stand next told parent wont abl travel public transport sort thing happen
go friend boy bu kept stare breast
catcal comment touch other area
touch
follow uncl own shop area
molest teacher class
fun park friend came man caught hand slap
man tri touch back walk street
survey carri safec red dot foundat along safec audit street market mumbai
go colleg bandra carter road mani colleg go student comment firl time even touch run bike happen almost everyday
incid occur aunt rajasthan holiday famili guy would continu stare shop freak littl
harass
near du mean see mix type crowd du student victim evil comment boy friend girl could give back boy
 11% |########                                                                |
crimin usual chain snatch comment variou bad activ
sexual harrasmemt dadar market
friend mine group ill manner guy comment obscen bodi
auto rickshaw driver pass comment
pictur taken near ashok vihar
stalk leh citi timeeven
teenag boy openli invit girl sex
frequent
2 men stand near tea stall right next hous keep stare whenev wash utensil tap outsid
ogl facial express
grope saket
friend return home work two men start whistl follow us scare walk faster ignor told mother told ignor avoid troubl take differ rout
happen morn time month ago
way shop man next ropad start call ignor
incid took place sector13 rohini friend go colleg morn two boy start comment us
come villag travel crowd bu boy touch girl inappropri mani time girl ignor later aunt came
walk friend guy start whistl us
man caught red hand defil 10 year old man taken polic station lock night includ 10 year old
walk man whistl look back made weird express ignor
mall elev liftman look like way lookinh felt violat know
friend touch breast boss md
fake id made name use send unnessari mg peopl face book
way school saw men follow ladi way work
teas group girl throw pebbl
follow
friend good time friend notic jerk take pictur make video
two girl get harras comment boy
near metro station
group guy sit cross road guy whistl
friend normal even walk came across two men start follow us pass inappropri comment shout help disappear bike
man wink shout futur wife
incid took place even strang middl age man kept follow facial express code conduct inappropri
comment
even 2 boy snatch chain
catcal comment touch sexual invit rajendra place metro afternoon shameless describ word
parent friend hous usual go alway area harass face wit mostli morn
file report 1 day ago larger buse hide gone still smaller van size park next build today roughli 9 pm thing call watchman ran away anyth could done could even get look face
sinc ranganathan street crowd street men occasion tri touch
travel bu colleg got seat sat bu crow sometim man enter constantli shake crotch touch arm
group person sit car follow rickshaw tri talk girl insid rickshaw pass loud comment
come back home guy stand tree constantli stare give wierd look
touch uncomfort
incid happen public vehicl countri suffer fuel crisi public vehicl pack sister travel micro bu space put leg suddenli saw old age man abt 5560 yr touch hip teenag girl sister say anyth man old man chronic touch hip harras girl notic man return back said noth
happen afternoon
friend mine comment upon return tution
guy masturb scandal
misbehav
realli bad
metro station guy came hit breast couldnt react
comment hauz kha metro station night time
touch
rain danc parti guy tri touch pass comment follow us even
even time bunch boy tri touch pass commentss
crowd bu happen friend sit gener seat left side bu man next friend sit aisl side felt someth shoulder turn check turn man erect rest shoulder happen best bu 296
happen maharani enclav night comment dirti comment
harass
night walk drop friend apart next minut know guy scooti hit ass drove
chain snatch najafgarh
bu street safe
bu conductor comment suddenli grip hand refus go start rebuk shout
happen afternoon
night boy comment
coloni boy came comment someth bad
friend travel bu alon guy sit front came side start come closer
uncl use show porn use touch 510
two three guy comment even travel
men said bad word comment figur even took pictur
stand station wait train arriv notic man brush felt uncomfort
facial express write mobil number unnecessarili train seat happen train rajasthan borivali station
person touch sexual organ public vehicl
boy touch girl bu soo uncomfort
boy use teas girl disturb chang hr rout
man came close friend move away start accus us public go place excus colleg
even 67pm sister touch travel rikshaw
boy snatch chain woman night 11 p amp incid becom common routin
around 5 even guy comment make facial express near rajiv chowk platform
pictur taken comment pass
incid took place time travel rickshaw two guy car follow start pass lewd comment even rickshaw puller turn rickshaw wrong direct follow cue
even 7 pm
climb train man grope unus place
return shop even guy came start talk tell getup other scare
harras guy bike
happen morn
bu boy tri touch n kiss cheek
sunset juhu tara road road juhu beach link road pass hotel sea princess becom flood call girl sit auto rickshaw await client room nearbi shanti whole set well plan sure polic conniv
6th grader teas 1st grader school bu lift skirt
get stare go school
man pretend drunk start touch girl privat part claim know
class 8 friend got harass last period classmat touch privat part intent got afraid share later complaint teacher
front aunt home small ground group boy gather smoke teas girl victim decid share problem aunt took action remov feel safe
ladi walk along riverroad go cosmet shop men seat along pevment start star shout suggest ignor provok way back told quotyou noth proud ofquot
beer bar close two girl hit men refus go
travel bu 724 person drunk amp made bad touch
walk street girl clap hand get attent blink eye
chain snatch 20th aug 2013 around 1030am rakshabandhan come nephew home two peopl came snatch chain woman
inappropri touch morn afternoon rush hour thane station
newspap seller keep stare everytim pass street pedder road near bank india ignor
go market car stop next comment cloth ask join
8 pm presum young girl come back school drope motocycl enter cite hostel caught rape three guy rumour rape refus date three boy
school boy sit street 2 girl 2 boy go street heard cheap comment quotarr pakkad le chodna mat chuml chumlequot catch let go kiss kiss
aunt go market boy follw
beat student teacher
two biker came bike guy sit behind snatch wallet
harass
guy follow took pictur
man touch girl hand move away caught hand grin
men call name made uncomfort walk
leav colleg guy comment
realli bad near arsd colleg
neighbor sister age around 23 sexual harass boy work doha qatar post nack unwant pronograph video fake fb id
friend way watch movi men whistl sing romant cheap song behind us
crowd newroad person pinch chest went away even notic
friend sit cousin read holi book teacher touch inappropri went sit place
walk road saw guy make bad comment girl pass
come back home guy stand tree constantli look give weird look clearli look mb constantli stare
stare
take pictur make video
sexual assault twice life age 4 14
harass even
group boy tri touch friend go market
incid took place around augustseptemb 2013 afternoon 34pm near bu stop opposit colleg vip pitampura passerbi ogl whistl
boy whistl walk footpath
boy follow girl hous tri touch
incid took place even insid dtc bu 883 friend man first tri touch back part friend bodi also expos privat part
bu go colleg sit aisl seat man start ruf crotch shoulder
person comment respect religion cast
 13% |#########                                                               |
morn friend go colleg ratnapark sky bridg encount man suddenli grab breast walk away noth happen
group 4 boy touch bad manner continu bu rout 721 753 821 853
touch grope najafgarh
group guy came start pass comment happen afternoon
old ladi walk suddenli 3 men came push snatch train slap arm ran away
man use follow us everi touch breast ran away
leav colleg morn guy start comment
incid happen bhatbhateni superstor saw girl age 20 approx follow old age man give troubl
man stand besid touch back hand also bad thing difficult describ narainar roadbu
realli terribl
teas girl come school
climb stair person touch butt felt back
touch grope morn
practic go home boy deliber caught hold hand
misbehav
person tottali unknown touch bodi neck waist
eve teas
come back work use laini saba ground rout man appear came straight statrt touch butt see peopl come ran fear life
afternoon three guy initi follow friend thn later saw click pictur know ask guy help us got us phone delet pictur taken
boy start touch girl hand girl gave boy look move sometim grab hand touch ass
guy brush elbow chest
saw woman touch breast forc man
boy whistl way school
friend boy comment talk nonsens felt uncomfort tri escap place soon could
entir stretch littl ahead movi time theatr ramchandra lane extens unsaf due absenc street light entir road dark fill pothol ad presenc liquor shop cigaratt shop lane attract young boy sit drink till late night lane also slum settlement make thing wors slum dweller add unsafeti road
ogl weird facial express man car stop walk road
inappropri behavior afternoon
saloon heard ladi discuss taken pictur without consent later saw pictur wassup group shock
near select citi walk autowala pass comment someth made cheap express happen even
boy known alway call girl whistl respond hurl insult
go school road quiet man came near tri ignor came near tri touch ear realiz tri snatch golden ear shout thief ran away site
gener
random guy touch friend breast ran away return home tuition
school brother friend stop knew stop tri kiss
boy class 3 abus infront school take cloth
guy keep take pictur friend without permiss everyday respond choos ignor
lost phone night crowd place
man touch way school
two guy pass lewd comment happen even
afternoon travel comment upon cloth wear
friend bu stop boy start comment
felt annoy unsaf incid afraid travel alon colleg
friend travel share auto guy sit next unpleas way
got stalk day 2 peopl
husband beat home even outsid alcohol work spend hard earn money alcohol go shop buy groceri make alleg affair complain polic sever time help tell sort live near wednesday market
unsaf pedestrian bridg connect borivali east west presenc drug addict bridg presenc anti social element bridg desert non peak hour like afternoon 10 pm light function find broken alcohol bottl cigarett condom lie earli morn
india lot gang rape go tell take action
catcal comment bad facial eexpress h block shastri nagar rape girl age 5 6 year
fate even three month ago left home go club way attack gang arm robber strip belt cell phone money evn jewelri left strand isol dark street didd know
girl alway sent man get paid tell girl sleep tell anyon
touch crowd cross
happen even
friend bu stop old man flash genit us walk though noth happen
50 last mani year almost everyday victim sexual assault travel unfortun never could take action far
incid took place dtc bu 883 morn peopl bu pass comment also tri find reason touch happen often
railway station
peopl crowd tri comment inadequ sister happen even
travel bu got stop start walk toward hous two guy come toward pass made cheap comment
touch night travel feel safe
even bu saw incid happen regard mobil snatch
colleagu mine emot pressur boss sexual intercours
use visit colleg daili vehicl an face comment indec exposur 600 quit riski travel bu
evenung man tri toucha girl
take benefit crowd mani peopl tri click pictur wear sare even
start campu sociocultur night campu call group girl ask give money refus later slap remov dress someth ugli
ogl facial express
happen outsid back gate ladi shri ram colleg lajpat nagar morn 10 01 2011 walk colleg got grope way
give taxi driver larg enough tip carri bag took upon decid grab right breast instead
friend wit incid local railway station crowd place wait realiz man stand behind tri ejacul dress behind moment realiz turn around hit hardli man push back ran away
group guy tri touch grope friend
guy look sing song way ti market
man neighbourhood kidnap girl rape victim said carri away dump
man help give lift rain later get kibera start touch thigh privat part
friend mine attack rape arm rubber happen come back church littl bit dark even men stand road sudden girl felt tap shoulder
come delhi dehradun night bu usual travel night save time request give seat side ladi unfortun guy came night around 3pm almost everyon bu asleep halfasleep felt hand approach inner thigh touch privat part immedi stood start shout peopl wake see creep sadli even move bit open eye noth even ask happn went conductor request chang seat final got conductor seat couldnt sleep rest night kept think mani day till today feel like give hard punch face
walk street boy comment hair style look
stalk way tuition
boy show video near church alway tri lure school girl sex
happen even around 3 month back way madangir guy follow
four boy tri take pictur friend come tuition
neighbour told follow give guava went plantat pull cloth rape
bad comment girl
harass
return back brother home teas group girl comment
friend stand somewher model town saw somebodi click pictur although sure click pictur express made quiet clear
harass
deboard metro realis man taken snap friend make indec express
stalk eve teas
wit husband beat wife public place happen afternoon
2 boy take pictur friend metro station turn away
girl wa chase follow sever boy around legion maria church area way shop around 7 30pm
friend walk road guy pass pass comment
regularli beaten husband especi use come home drunk
sister come back studi saturday even boy call ask could talk sinc late 6pm refus boy caught want rape peopl came aid
happen bar night stupid guy tri touch slap instead
misbehav
sexual harass parent away left hous uncl night sleep uncl harass saxual thank god struggl let go ever think struggl make loos respect report case mother drove
bu old man touch comment beauti ask go dear
incid took place even friend sister market follow boy ask friend
age 16 yearstyp harassmentcatcal school leh 2015 time comment school leh 2015 time
gang peopl whistl friend way home
 14% |##########                                                              |
twice grope drunk men societi night hour school know react
boy comment girl
incid took place afternoon stare inappropri manner
way buy stuff suddenli man pretend mad walk pass ran woman touch butt laugh ladi felt bad took
abus famili member home
friend sexual harres uncl
plan buy sock even time shop close saw woolen shop thought could get pair sock ask buy pair sock said look beauti want make friend ask hug refus want go gave gold chain tri persuad sex came told friend came threaten guy
step offic buy munch stupid roadsid stranger start pass comment like ownaman leke kidhar ka rahi ho
happen morn
eve teas boy watch film
touch grope comment rajiv chowk metro station
categori select afternoon even
touch board train
night time lot guy saw comment upon ignor walk away
way son class right outsid build hail rickshaw two teenag live build start point made comment laugh ignor second time walk question yell
walk bhatbateni supermarket saw boy teas girl pass
stalk stare grope sexual invit
man tightli grab thigh pack micro bu
go across street groceri store two men began follow bike comment
usual avoid comment guy use follow everyday complain teacher help
walk road man pass say want touch breast could anyth
ladi wear gold chain snatch two guy happen even around 730 pm
happen vacat famili shop guy tri touch got scare
bike rider tri stop stop haul insult sayingquot bent leg dumb uglyquot
old woman walk street two boy bike put mask face snatch chain
wait bu unknown person ask whether want lift car spoke dirti tone start click photo car number plate fled away
use chang metro around 8 even mandi hous metro station take metro toward ito wait metro realiz somebodi stare long took serious ignor suddenli around 1520day walk toward platform time saw person look smile wierd still ignor continu 45 day use catch metro mine time even tri talk ito metro station use exit happen end chang rout
around 1012 year someon tri put hand insid back pocket crowd
go home school way home guy comment girl
take pictur comment rude behaviour catcal
guy car stop bu stop call friend ask direct masturb could realis happen drove
walk toward stage guy told av got beauti figur
unknown boy came ask name nd ask sit time
whenev drive area night usual get follow guy car tail car tri block way
near park walk
men stand backroad ogl place low lit night intimid walk
along friend head toward gtb metro station rajiv chowk huge crowd men tri touch posterior part bodi
catcal riksha driver
walk street broad daylight man bike stop ride came close sang pervert song quotzara zara touch mequot
survey carri safec red dot foundat along safeti audit street market mumbai
comment catcal group drunk rickshawpul
man touch breast tightli could anyth felt much pain
12 birthday way back home dinner mother uncl aunt cousin cross road man came touch butt later turn whistl first experi sexual harass
stand line group guy disobey queue scold start teas us happen around 315 afternoon
feel mistreat buse
quotat uhuru park town friend shout street boysquot
two girl rape onknown men night
travel buse difficult especi rush hour
month ago walk n wear short dress guy ther commment sexi fee same
buse head place safe
sexual harass accord friend said boy pass came took without know happeningto could realis boy boy slept boy took book gave told mother ask say went beg book studi took back home start shout faint taken hospit whole stori came open stay camp idenau boy stay block 20 idenau said boy use black magic
go home school notic brother look backsid boy pass bad comment brother threaten still comment use bad word n rough voic
830 pm go baluwatar near nagpokhari two boy tri take pictur said policeman walk
six boy 2 scooti eve teas friend street silent
catcal whistl comment especi metro station
man told girl go see want touch
realli unsaf area women
man touch privat part travel bu
chain snatch comment outsid ganga ram hospit
man came run toward knew happen grab phone talk ran away
grope catcal kandivali railway station
harass
age 17 yearstyp harassmentstaringogl roam leh market even stalk 2014 kacho sermon choglamsar leh time catcal market 2015 tuition time time comment moti fat time
boy follow girl alon road start comment wistl
happen near liberti cinema hall around 7pm neighbor roam front home two biker tri snatch chain
molest attempt rape
saw girl teas badli boy girl afraid even follow
person continu stare amp make comment look
biker stop front friend ask accompani
bike rider cycl rider comment run away
call name like darl chuli apricot
crowd train man tri pull skirt touch privat
happen even
even travel comment feel like mandatori thing
way home met boy roadsid saw whistl friend call name
coach classmat use follow us everywher use go pass comment dress bodi languag felt uncomfort
parti danc colleg guy pinch girl inappropri part
teacher beat student
cousin scooter ride along marin drive sea face bike came along behind 2 guy kept smile ogl kept tri distract us would zoom across come way happen marin drive follow us way home lost half rout stop moham ali rd divert rout
quoti feel strong hatr toward ladi dont knw quot
lawyer work late almost everi second transit new bombay andheri via wadala rd station see local boy boy travel line pass expressli lewd comment gestur toward femal commut transgend commut effemin men within train station full view polic offic deploy station
night time friend comment drunkard got scare ran away
walk catch train amidst throng peopl wale snail like pace suddenli man besid make way legit brush hand breast complet awar grab hand push away
dtc buse green line safe
cheap comment follow click pictur
okay incid happen last year e 2015 finish class vashi head toward vashi depot take bu place kharghar navi mumbai board 504 number bu took ladi seat window side phone call minut realis touch near boob behind instanc thought might rod seat later realis somebodi hand tri grab caught hold man hand right turn turn abus ask upto quit shock sure expect girl react way men think scare speak way fear bu conductor came site pull shirt put bu well later 5 minut man sat besid start touch thigh burst even madli 5 minut back first happen turn toward without give second thought liter slap abus angri angri happen peopl bu hit even put happen hesit whatev necessari thought miseri end anoth lecher stare time complain conductor forc bu
hotel boudha sidharth hallopposit sister went enjoy eat food old man start look himm leav stare
teacher beat student give sister phone number touch bodi cane sexual relat ship minor
fate monday last year small villag call bova victim young girl stroll suddenli guy came hold knife ask girl pull cloth els would forc girl refus guy hold close innoc girl shout rape young guy night
group men whistl way church decid ignor
pass ratnapark father offic guy stare facial express disgust look angrili went away
go home boy harass touch breast girl could anyth
touch grope
buse head noida
go male friend even group boy start whistl comment
group guy tri push crowd
go fetch water near home white man came pull touch breast manag escap
travel bu guy touch uncomfort way twice happen ask stop
catcal public transport
return coach around 6 pm two boy kavi nagar bike pass comment firstli ignor start follow continu ask person detail refus tell took mobil phone pocket went 23 day continu stalk
purpos make nois attract attent annoy
boy alway follow girl home without know
even walk friend friend comment upon
man stare
lot comment whistl travel even
stare whistl wear short kalkaji
illeg pedicab cyclist harass non local pedestrian attempt imit speech
dinner meet uncl call room touch inappropri fled
86 year old mother attack chain snatcher build 3 decemb 2012 scuffl fell fractur thigh bone
 16% |###########                                                             |
comment ogl mean nonverb verbal sexual harass common thing everi girl ignor solut rememb kind thing mention possibl common
biker snatch chain
boy walk stare girl also comment happen even
go fetch water boy whistl ignor
gurl near matern uncl hous harass boy grp girl remain quit took action result gang rape 3 boy die hospit sad news whole societi shameless act
man stand near tri touch pretend check pocket exchang seat friend went littl bit far
inappropri danger behaviour
man cought sex young girl call friend told loc door outsid
guy constantli stalk stare women pass obscen comment ccd connaught place
work hr night journey feel unsaf
15 year old boy year ago stalk group peopl rob cell phone money etc
alon rent room husband night duti somebodi knew person threw stone door scare next told landlord
incid happen near shiv mandir
messag rough word facebook
two friend walk street teas drunk men suddenli caught hand said like polic men came complain blame back us
go friend place notic guy stand close stare inappropri make indec gestur
night sleep
cousin came meal home meal talk touch bodi unnecessarili like invit sex
man near hous would keep stare start follow school came realli close start whisper sexual suggest thing even tri touch inappropri go school day scare inform mother also happen near hanuman mandir vika nagar
ass grab
comment upon boy disturb
friend school teas bull sing make feel uncomfort
boy invit us differ place say vulgur word
night street light make area realli unsaf unsaf
male taken pictur girl know kind harass peopl shout share stori famili
harass
23 boy pass comment walk road
age 17 yearstyp harassmentcatcal neymol market near bu station leh march 2016 even
took place afternoon wit harass delhi mani time mani incid actual describ action taken male around irrit part
travel scooti boy scooti comment
1 went consult meet upcom ad shoot radio jockey quit famou attempt rape let go scream top voic learnt lesson never go consult alon live realli horribl world 20 year old walk place scare tear run face next look back want sue realiz wit proof given polic time get work sudden think could kill ruin life famili member would bring differ five year line move life still time wonder mani girl done rememb exact date month 2 also incid gr fantasi park bangalor year 2011 water park area friend went insid water tri swim someon grope breast 23 time continu good swim could anyth seen guy face time got water angri hurt disgust nowher seen went home straight right away alreadi upset 3 engin colleg interior place friend return movi around 7pm unfortun streetlight connect road colleg bunch guy came bike slap bottom girl whoever could reach dark bike fast noth much could happen almost everyon hostel get alert go side road soon bike pass us
secondari school claasmat use come closer start make unecessari advanc like touch kiss without consent
board train sion thane got seat notic thr gap creat haphazard stand peopl guy take pictur girl front partial open decid confront drop next station
ill treat ladi
walk friend man came bike said bad word friend
whistl group men
elderli gentleman wife mug st franci rd today robber 3 men wait auto rob gentleman gold bracelet wife purs
transgend ask money auto stop signal refus start touch privat part
man beat wife neighbour call polic came rescu
unknown guy abus girl touch unwantedli
ex boyfriend forc get back start touch breast
shop right opposit build shopkeep follow go class bu jogeshwari bu stop bu till class thing ake reason amp answer stupid said came drop amp use wrong word get late rush wait class see told sir amp came along see ran away amp stop follow
coloni boy start comment left
realli bad
pictur taken outsid colleg afternoon
realli bad
man grope privat part
wallet mobil stolen someon bu stand
harass
way school morn ti boy follow tri show pron video pictur disturb
happen someon know around 6 month back friend stare school sarvodaya vidalaya sweeper happen morn
stand bu station guy tri feel happen afternoon
catcal whistl 4 month ago even block pitampura women face whistl comment
ogl catcal whistl facial express
teenag girl molest group boy wear tight skirt refus advanc indec touch harras assault
dtc bu 764 nehru place new delhi peopl use touch privat part
travel girl friend munnar bu halt somewher munnar 10 min break roam take pictur suddenli man expos dk start wave toward us embarrass move ahead month date locat approxim
chain snatch sexual invit
person stalk day happen time
even wit chain snatch incid
disturb hear guy comment
famili friend peep gap window chang
guy pass comment cross start laugh us
walk street boy comment
behind richshaw boyfriend driver kind fun want tri rickshaw want insist want nice accept sat close grab moment touch pector manner quoth may thought go accept teach mequot kurta noth reveal terribl
incid chain snatch happen yamuna bank area file fir
come chennai sit railway station wait friend pick random man sat next start watch porn cellphon keep could avoid see
outsid colleg come across cheap creepi peopl
girl walk near makina stage cyclist around start comment hip breast
two men bike snatch woman chain cross road
touch grope push deliber
even comment touch common area
come home school brother afternoon guy car stop car stare badli blew whistl
boy stare girl laugh worn dress short
men sit around kamkunji area tell us would give us money slept tell quotnoquot insult us say wil age watch
bad stare whistl pass street
boy incid even boy touch organ breast girl girl feel uncomfort slap boy
went visit friend mine usual know mind long greatest surpris start touch unexpect part point strip rape though end date disvirgin
continu comment travel even hour
harass local train
stare peopl pass king cirlc station tothrow stone acid pass comment mahim station
english teacher found harass young boy stay privat hostel
woman whenenv pass near school call love tri touch privat part
whistl
peopl call chinki abus north eastern horrifi
sister went visit certain boy boy want sex sister refus found guy drug put sister drink cought sister ran away came home told stori
40 year old man rape 7 year old girl shiv sena maidan sanjay nagar kandivali west mumbai
time bandh andhra pradesh sister walk street light boy ride motorbik tri touch us start comment
girl bu n group boy tri comment girl n sing song
grope shahpurjat
wistl comment bad word road walk
masturb park
group boy sit car pass lewd comment
comment
school name jp high school health teacher use rub back time use take hand waist line time know harass even though talk femal teacher also anyth
ogl grope
walk road man came near grab breast slap ran away
bu go hous return sundhara shop way back man tri touch bodi part felt uncomfort share mom
23rd dec xma eve friend left hous meet uncl go enter motor cycl anoth man sat behind man alight girl wait motor cycl start move felt someon hit behind neck collaps next morn woman saw took hospit doctor said rape
friend travel public bu colleg harrash boy bu boy tri touch n feel uncomfort n use safti pin n poke ran awyfrom side
sexual harass auto driver auto way colleg pass comment tri touch catch hand dirtili ask stop auto scream refus even took place juhu beach bandra hill road
summer 2011 stay manju ka tilla return dharamshala foreign travel alon would alway stay guest hous sinc area felt safe morn 9am prepar leav guesthous take train agra corner ear cleaner happen suddenli offer clean ear said push wall side alley main street proceed touch breast nippl somehow push away race back guesthous guy recept ran tri find ear cleaner gone shock shaken upset train catch head new delhi train station taxi station driver call guy help carri bag agre much would pay meant help find train reach train went 2ac section travel time empti still put bag extend hand ask money said reach agre sum time start feel thigh move yell back told back get shook head said ok money gave rupe grab hand put erect peni deepli disturb luckili tri free hand grip peopl came train ran opposit end 11 peopl travel comfort made sure arriv safe agra worst day life
 17% |############                                                            |
catcal hour
incid took place bu rout 33 noidabhajanpura bu man staand behind fall uncomfort manner
invit place friend thought go got start say thing start touch refus want forc himelf start cri let go
pedicab cyclist station near red chair yell bayot pedestrian
travel bu guy kept touch harass wrong way
survey carri safec red dot foundat along safeti audit street market mumbai
got issu known senior colleg adhoc professor use ask girl student accept friend request fb sometim go coffe ask come alon cabin discuss thing
walk man pass comment tri touch walk faster
incid touhc grope
man click pictur friend tri catch hold fled
touch wrong way crowd bu
walk ill treat
friend walk toward tuition enter build man came behind grope
teenag girl go toward metro station suddenli car came near boy sit car open window gaze toward awkward manner said quotlet go quot
tri touch morn
man unzip wave peanuts peni crossroad main road plain view
eve teasingno street light
way music festiv saw man remov privat part squeez
sexual invit guy came play cricket local afternoon
hello sir madam mere naam jagdish sharma h apn bhai ki help karna chahta hu mera bhai mujh se bada h uski shaadi ho gai h 3 saal pehl 2year 6 month pehl vo hame chhode k kiray pr rahn chale gay h mera bhai hendicap h 60 meri bhabi usko marti pitti h hums nahi mile deti hume phone bhi kre bahut ladai hoti h meri bhabi ke sabhi ghar wale mere bhai ko paresan krte h mera bhai bahut paresan h time unki maddat k liy gaya meri bhabi ne road par hi tamasa kr diya kud hi apn baaal kapd fad kr mujh hi rape case fasana chahti thi polic bhi le gay mere bhai ki koi bhi help nhi kr pa raha h pleas app se hath jod kar vinti krta hu ki mere bhai ko us dusto se bacha lo plz plz plz nahi na jane mera bhai kuchh kr na le vo ladki bahut hi drama krti h pure din ladai wali novel padati rahti h samjh nahi aa raha kese apn bhai ki help kro polic bhi nahi sunti galat koi bhi ho par ladk ko hi galat bola jata h ap bhi kya sirf ladkiyo ki help kart ho plz plz plz sir help cont 08826862360
realli bad
misbehav
travel even particular place got lot indec stare men women
walk 15th main turn 16th cross hsr layout sector 4 heard sound scooter behind move left hand side road particularli short stretch littl dark hous either side road tree tall thickli grown hide light street lamp coupl minut realis scooter overtaken realis turn head see scooter rider put left hand grab right breast wear thick sweater late januari cold even even felt complet violat scream abus fortun sound actual came throat realis shout grab breast taken immedi heard abus loudli stop turn made return moment young man turn corner think made take fast thought would turn get 17th cross turn went road disappear still rememb vagu face fact old scooter vespa lambretta wear jacket month scare everytim walk road heard vehicl behind howev gradual support friend famili prayer overcam professor train peopl report crime resist fight gender discrimin middl age woman perceiv strong confid shatter incid like first time someth like happen long time let guard sabbat time rare went report polic know would tell vehicl number definit descript crimin wish report someth like happen may made polic alert
movi earli thirti guy tri touch hand
man age around 30 follow
chain snatch park even coupl boy snatch chain handbag park
travel crowd bu middl age man 3035 touch privat part scare could speak word
humili
met man run girl rape coridor girl cri help
chain snatch comment bu even
men boy hang around outsid station stare pass comment women passingbi
even stare
comment
two grown men father famili road side wait friend touch breast got confus even understand happen
walk road night suddenli felt someon hand pat butt scream bicycl ran away fear action passerbi
ask husband accompani approach road station safe
return colleg guy continu stare commnet touch
talk objection pictur friend bu
go home school stupid guy call tri snatch watch luckili ran away
follow sinc 2 day
mother sent go call friend went found brother hous ask sit wait sinc far stood went outsid came back close door sat next start touch
survey carri safec red dot foundat along safec audit street market mumbai
presenc roadsid romeo near saidev hotel due lack polic patrol make women uncomfort shop visit beauti salon
cafe younger sister group boy stare later actual tri click pictur mobil camera
even happen
girl age 18yr rape fellow classmat cut leg later found dump along river
follow
around 230pm boy stand near tree use foul languag abus thing
move toward home saw boy stare give comment dress
come back offic boy whistl around make feel unsecur n even follow
girl teas boy gongabu area area alway crowd boy never miss chanc teas girl
walk around street saw boy teas young girl
whistl lane commun
train travel varanasi jabalpur night travel alon
report allow understand girl forc commit suicid might see tv news day girl commit suicid blackmail sexual harass stori 15 gone visit friend street notic guy stare 18 19 even wink whistl comment ignor pace friend place told incid mad want go confront guy forgot talk rest afternoon even total shock saw guy enter friend hous introduc brother speechless guy grin dirtili went room friend got dress want tri show mood disgust want shout brother tell friend kept quiet succumb pressur went chang dress show dress go back chang origin notic stand near staircas smile evil way block way held hand forc see video horrifi saw video chang cloth toilet start cri beg delet said sex moment plead beg know got strength slap friend heard nois came told everyth speechless know react cri said mani thing brother thing never neither
go colleg crow vehicl man lean touch inappropri
harass
even comment friend
assault three men bu go sunauli near nepal border varanassi follow three men onto bu saunlai bu stand verbal physic attack spit push etc fear life thought go drag bu rape murder exagger bu full passeng scream cri loud would even look scream bu driver stop bu ignor time man anoth foreign got onto bu stood made 3 men harass assault leav bu
saw girl touch aimlessli menwho smoke bang
go home met boy forc small girl give kiss
metro move girl compart guy saw whistl
classroom insid colleg amiti school engin technolog e2 2nd floor stand class wait next class begin male classmat walk fell top grab squeez breast nobodi seem notic embarass walk away stare assail face tri make sens happen happen
chain snatch ladi stand bu stand afternoon
walk pass bridg man call start remov privat part refus go start abus use dirti languag
return school teas group boy scare came home fast could
harass play basketbal boy girl play
morn metro station fad group guy make facial express ogl
cbd belapur taxi stand outsid railway station goon wander night
home molest hindi tutor made sit massag peni 13
harass biker
receiv unwant sexual attent hous parti man extend rel could even react
continu stare touch
gone friend birthday dinner hkv around 10 return group 3 4 boy start follow us start talk randomli ask us thing like name fb id snapchat id even told v r least interest bother backoff indtead got car start follow cab even realis much happen almost 3 4 time around hkv
bu touch inappropri man got immedi
go sister place guy follow tri chang path didnt stop follow start run n ran fastli took narrow path went sister place share sister said ignor
girl friend travel alon bu old man made weird facial express give sexual invit well touch uncomfort scare
walk friend certain man start take pictur us
winter season come home even time alon get vehicl plan walk came home short rout boy follow scare wait someon come way would go thank god came anoth man cross way
mahalaxmi way worli naka quiet lone peopl even bu stop taxi driver stand peopl walk even 8 road
friend mine femal got harass whilst jog earli morn
grope
teacher beat student
stare pass friend
come back tuition boy tri touch
shop met old man start talk man promis buy thing even tri touch ran
bu comment spaek bad word tri touch
public vehicl guy stand behind scroll hand bodi
wit incid chain snatch near red fort
sexual assault forc
go colleg even guy stand group constantli stare smile
saw girl wear half pant walk home street boy tell rubbi word kid catcal saw home street
realli bad
guy rub butt ran away crowd could get hold
road inner road subway dhobichaur unsaf even walk even time guy invit sex
walk toward hostel two guy made weird facial express inappropri
incid took place afternoon around 240pm 442008 mt brother come back home mother school bu stop two boy came bike snatch chain
afternoon girl pass metro station azadpur small boy came behind snatch phone
girl go school sunday boy comment pass depress went home rather go school
small boy nearli 5 6 year roam near home given realli fierc look uncl stand made run form abus reason boy could help ran away
guy slap ladi back wasnt good way ladi complain guy slap start call name
happen market went buy boy hit buttock other whistl name want buy even pull forc
wit incid chain snatch ladi two boy bike happen night hour
ass pinch guy twice within span 3 min wait rickshaw kurla station right opposit bu stand
old man comment dress
2 girl metro station go toward huda citi centr gener compart man touch like hell someon hurt sentiment girl
sexual harass went male friend hous cours discuss start kiss actual make advanc toward ignor close friend cours acess bodi tri resist lock door told give shout embarass us
amrket boy pass comment
stare amp grope
first colleg statin look indic suddenli man come touch realli bad way made realli bad nois
guy tea shop show facial express lunch break go shop guy work bank compani come ther sit comment us show dirti express
shock old man touch privat part
whistl chain snatch ogl take pictur
guy pass next made sure rob elbow breast guess effici way guy touch someon girl wonder quotdid realli purpos imagin itquot
saw men take photo woman neighbour
stalk autorickshaw driver
walk pass group boy way market start hear boy auter insol word approach felt hand boy buttock could react scare
2 men continu follow polic station highli requir
men usual call us pass darajani
 19% |#############                                                           |
2 clock noon head class chang two buse reach class hurri moreov new citi got bu bu stop bastard walk opposit direct came near utter someth touch front public ridicul cant believ happen terribl shock
friend talk said street pull skirt ran toward hous quickli realli cri talk
group guy chase ogl
comment near rohini west metro station even
jog morn face catcal men park disgust
even 6 pm
continu comment stare stand near metro station underw afternoon
experi girl caught way home club night girl rape group men bush incompet help indec dress say girl rape men took stick insert vagina die miser
realli terribl
stalk street light
two boy stand centr market stalk friend
night board train jammu guy guess worker start misbehav
case male teacher tri offer better mark colleg also case men masturb public
everi time pass street creep peopl especi young guy pass comment ogl whistl take pictur
incid took place night sleep upper deck person next touch appropri place
afternoon guy take friend pictur
chepirog rod hospit
comment touch grope take pictur sexual invit
alreadi dark someon grab sister chain
travel metro station colleg boy street comment dress shame full terrifi
kakariya today boy behav badli girl need lot secur
catcal pictur
harass afternoon
road hometown sunset dark street light chanc geti rob high sisit attempt rob harras
follow old man kept ask direct excus
come offic bu saw boy physic misbehav girl say anyth look scare
night friend told take shop way mate boy admir friend long long tht boy told friend want take scarf could feel better friend refus took friend chain said whenev feel like friend stare kiss chain
touch wrong way
last month saw girl eve teas group boy tri stop start argu amp abus marathi luckili amp girl abl escap
man follow platforn 3 ticket counter touch thirc
wait brother outsid colleg group boy also come look start comment bodi could control stare gave bad look apolog left
fourteen year old boy sodomis eleven year old brother share bed without parent knowledg
woman snatch person amp assault sexual
pass boudha street unknown man pass touch sister privat part
group boy take photo western dress girl
bu boy sat next first ask place suddenli held hand ask time shock scold
teacher beat pupil
hawker sunday market misbehav friend pass comment cloth say invit troubl dress like
go cousin marriag
go colleg man continu stare
go school dda park man start walk next kept whistl happen earli morn safe time
man grap boob
twice 18 almost got rape 1st standard music teacher second time 9th standard tuition teacher mba man follow stop scooti middl road drag atm forc give money though shout busi road came help
sometim ago travel autorickshaw dehli wear dress natur calf uncov men bike even car kept peep insid autorickshaw see althrough wich humil creepi reaction tri ignor look phone
saturday morn ladi wrongli accus go date husband due fact man use carri car go school howev man intent date refus
sexual assault realli small understand anyth time
guy comment nonsens
travel dtc bu 711 person badli drunk amp comment upon also made bad touch
two boy misbehav friend
man came close neck wave hand toward inhal deepli smile sleazi smile mutter someth hindi tone felt derogatori sound kind weird found violat behaviour experienc india
man roll paper threw whistl saw said sorri sorri mistak threw paper
happen past sexual harass metro man start stalk badli feel bad
four friend shop three stranger whistl comment us ignor first stop scold
2 boy 1617 year old way back school notic walk road decid fun idea walk behind casual convers watch attract butt move
met boy said love
test
return home school man bike snatch bag backward
men smoke bhang strate whistl uncomfort
comment whistl travel even
unknown girl hostel area got rape night four boy
seen girl teas stand bu stop railway station
recent colleagu abus phone colleagu came know report union
go back home guy comment night hour 1000 pm
watchman anoth school click pictur friend
order food room young man came insid tri touch pector arm understood quickli act accordlinli mean taht chanc thing boyfriend rest fellow act discust way
went visit sick friend hous found five boy room sat sat boy close door rape friend
two guy pass sexual invit along stalk
come mhada coloni harass
10 year old man 30 year invit store bought bonbon took near bush ask undress start cri forc peni vagina pain tell bodi
return home saw boy stare start comment follow till market
girl pass even around 5 rickshaw puller start comment abus whistl girl
ill treat
way school guy start comment touch
friend mine return back work guy tri snatch bag fail tri tyo grope fail attempt push road fell road head hit stone bleed man easili ran past hoysala van park end road
pass comment
friend rape tuition teacher
catcal comment ogl area afternoon
usual get blank call messag
comment
explain
man comment cloth outsid build loud everyon around stare
board train person pass cheap comment
happen friend mine home cousin step brother along harass whenev bodi home succeed sleep led fight badli beaten cousin step brother could tell bodi knew bodi would belief
touch grope sever time 3rd std 5th std day ago
go lodg fir nearest polic station due misplac id way guy came held hand went
walk along olymp road met boy said quot riper mangoquot
comment touch grope
bunch guy drunk comment cross road near connaught place happen even
got train man pass felt thigh
everyday stand balconi peopl stare way caus discomfort
get back home around 9 pm guy came behind start touch person part happen balaji chowk
time travel vehicl everi time peopl like 40 tri push girl touch hand
man use bad word us return home colleg
eye check optician tri touch neck
whenev go outsid hostel face kind harass everywher everi time happen everywher outsid colleg hostel
mother sent friend place reach find found husband told wait remov cloth put towel an came close kiss
honor killingmuslim girl want marri hindu man
sister catcal market
friend travel metro gener compart sit ladi seat middl age man came stood next kept stare make feel uncomfort metro yellow line
grope
servic road lead junction extrem dimli lit antisoci element often seen loiter around area
walk along road ask need lift follow anoth car night crawl lack respons gave made give
walk past store notic man look toward make express comment return
drunk peopl car comment friend
man pat friend walk near park dark ran away bicycl
colleg enter bu old man 70 touch back bra strap basic
grama panchayat impos public nuisanc case file polic station need go mri scan rel upsana hospit need chang cloth person open door mri scan ill person door close forget lock person hospit treatment auto rickshaw open purpos old men stand next fill crowd know monkey act shown public action action
pass immor comment teas took place even dwarka metro station
rape bu
man felt privat part behind
stalk commun indira nagar forc chang rout
friend atrciti late night coff starbuck came around 10 15 two guy sit bike start whistl start walk ask us could drop us home
guy tri steal purs backpoket scold
street show crowd mom walk suddenli felt someon pinch back part look back none
travel bu man mid age tri touch stare quit long time
friend gone shop group boy start whistl happen 20th septemb afternoon
due street lightn incid like chain snatch comment whistl common area
harass
two men bike walk footpath snatch chain
guy tri make physic contact
friend sent shop mom way mate drunkad man man want touch privat part strugl man touch breast
go school took micro bu found tri snap pictur uncomfort cover face mask
near colleg lakshmibai happn
 20% |##############                                                          |
guy tri grab hand friend forcibl dakshinpuri hblock
eve teas stare
trip chandigarh group boy click pictur
friend walk english tuition man scooter flash
brother law sexual abus sister law rape threaten
comment afternoon hour
raini season mani peopl around someon whistl make sexual invit seem odd somehow manag escap
man car approach offer car ride sat sever hundr meter ahead ask suck dick declin drop desert patch
night girl enter hous rape girl
public vehicl look window man enter sat besid plenti empti seat sat besid lose could feel breath hear comment beauti
teacher beat pupil
happen cinema hall even night
show privat part femal
men walk station sky walk touch walk also touch elbow chest parla station
cousin rape bridg 42 night go shop two men
afternoon whistl comment upon
insid metro male stare privat part girl feel uncomfort
hello 17 yr old girl man follow tuition class home daili scare share anyon u help regard
friend mother vikaspuri f district park saw lewd comment pass boy stand tri ignor went away alon could give back
holiday friend went visit pashupatinath templ stranger flirt anh tri catch hand frighten
video room around area ponograph movi shown ayon entranc fee regardless age
kamla market four boy start follow pass comment
survey carri safec red dot foundat along safec audit street market mumbai
man continu whistl pass comment walk
chain snatch afternoon
seen oneday metro guy click photo girl
road isol road tea shop given find differ gang boy market peopl sit throughout pass comment girl pass use back gate princ arcad build 2008 girl call polic control room constabl appear ask girl would file complaint write would someth even dare go near bunch boy disgust stretch men stare pass comment long
group guy car start follow abus
bank go back home man acroo street car call went reach touch butt ask go afraid ran home
two men follow aunt sunday morn tri call ask come nearbi park auto someon knew wait
friend enter area light badli construct road
vehicl pack women sit sit next complain man side frequent touch arm knowingli
peopl keep pass comment whenev go home market
comment afternoon guy comment ask could drop somewher
travel best bu 9th std elderli person tri touch breast actual use forc creepi disgust
poor street light
basic elder men time tend touch grope especi crowd bu happen seen happen other well
victim subject fear sudden boy stand side bu stop made obscen facial express
afternoon 2 pm
middl age man stalk made indec express gestur chhatarpur metro station 3 pm work dress formal offic attir indian report kept put hand insid coat pocket talk phone
friend sexual abus father father come close famili member home
peopl tri follow scooti go back home
man touch street pass across
guy tri touch make differ facial express around even
mobil got snatch abus
boy wistlind comment tri touch bodi part
watch porn wait watch footbal match apocalipto
go coach 23 boy comment
catcal
whenev go parent place observ men bike alway follow ever go know whether intent scare
near shivaji market pushp vihar new delhi someon stalk
transgend sexual harass men train
walk olymp road men idl outsid movi shop stare girl pass pass well pstare
someon tri becom physic distract steal purs
bu conductor refus ticket femal passeng
within famili brother rape sister blood relat boy took advantag sister n rape
unfortun incid happen front sobo mall car start follow us
two unruli men white maruti swift hr 26 cd 8118 first revers car red light traffic stock still even repeatedli honk stop light chang manoeuvr 2 car ahead driver show middl finger amidst raucou laughter within car light chang busi intersect could hail traffic cop
go auto suddenli car 56 guy put realli loud music disgust cheap song make weird express follow us around 1520 minut realli scari
man tri touch breast ass stand pack bu
mathura highway enough light way night safe individu journey alon
stalk
ramp walk
heard friend left school went get marri
touch grope rape sexual assault
pass friend way met boy sit roadsid start call us refus go
walk road boy comment tri take pictur
went grind spice cook guy even know came held tight complain gave kiss
certain man told sexi thigh would like kiss stranger
work school colleagu pass unwant comment
gone sunday market tee hazari guy follow market look inappropri made indec gestur kept ignor also reluct tell mom guy kept act final lost cool went shout make uncomfort start lie peopl around gather start hit
real brother sexual abus told mother refus believ live near wednesday market
hear guy comment even
harass
stalk drunk man
go shahdara metro station way man start whistl pass cheap comment metro peopl touch pretend noth happen
dubai wonder countri men mean harsh toward women got horrifi polic men rude
ocassion seduc dad time tri rape scream scream dad tie mouth
boy comment wait bu afternoon hour
aunt walk crowd street guy actual open zip pant start rub peni ejacul back
misbehav
sent shop boy sit outsid call refus answer start talk badli
eve teas colleg student boy street
travel bu saw man tri touch women
waa sunday morn go church karanja stage wait toboard vehicl man stand behind snatch golden chain peopl wo say anyth felt scare screem watch walk away
misbehav
happen morn guy whistl call name took place move bu
home group femal friend came across groiup men look us laugh made us feel aw
around 1030 pm suppos receiv femal friend soon came group 78 grown men start pass comment
survey carri safec red dot foundat along safec audit street market mumbai
bu station old man start talk start touch hair
refus get rickshaw connaught place new delhi rickshaw driver follow foot 30 minut despit tell repeatedli knew go could leav etc final ask quit firmli leav alon start ask money ask 20 time amount quot rickshaw journey want anyway would give money start shout swear say would throw stone put someth anatomi quit catch said think guess f mouth peopl stop man think twenti particular help get rid rickshaw driver
random man bike stop school go girl flash us laugh ran away
friend catcal 2 men near bu stop
guy comment
boy kept come street hous would sing sexual suggest song whistl whenev pass
way back 2006 guess mom wait bu stop around 730 pm dark bu came catch idiot almost tri pinch near breast miss instead got shoulder kid back moment didnt realiz happen move get bu day carri pepper spray key chain everytim go happen would probabl kick ass spray eye even realiz hmm
incid took place kashmer gate metro station yellow line even girl sit metro guy stand next constantli stare indec manner
student india teas du student
man began comment loudli dress
ladi grope right thane station daughter handbag snatch grope push side platform 1 thane station
way visit friend came across boy quiet simpli greet continu journey walk call told rush see someon suddenli ran held waist tri kiss struggl gave kick final ran away
guy stalk contin
home alon cousion call ask alon home suddenli sister came compani saw ran away without enter home
go colleg boy start sing sexi ladi floor offend
 22% |################                                                        |
colleg fest invit muslim friend usual dress quiet simpli ordinari also wear burkha complain boy pass indec comment whistl
go home work biker tri tp rob
nepal telecom pay bill sister felt urg go toilet ask guard toilet instruct toilet came stand tri catch later complain polic arrest
sexual harass public place
boy flash privat disgust young
go watch movi hall boy pass comment us
walk certain boy start tell love boob want touch
catcal main road
guy kept hand waist start say pathet thing
old man expos privat part near bu stop pass dirti comment
two boy follow return back kathmandu kept talk irrit
stalk touch grop
guy cycl comment indec languag happen near ramja colleg morn
friend uncl use touch inappropri keep pictur phone tab
alway notic thing area even night time mostli
comment
friend near ito bu stop someon comment awkward manner
incid happen twice insid auto halt signal near borivali station right sky walk unabl see men time would swiftli move signal open cross road quickli hit breast woman seat outer side auto much done auto would move away signal green strand hurt humili
continu comment step even
night guy bike follow girl rickshaw pass vulgar comment
sunday morn go church met group boy call gaza start chase us
wear short peopl stare also whistl
travel train friend vile parl grant road got male compart man continu ogl us even comment realli annoy
drunk guy start follow shop stand
boy call use bad word walk road
comment pass lane commun chang rout
friend trust much sodomois class 5 pupil highris reort
earli morn group 3 girl
happen amar coloni lajpat nagar winter 2012 night two peopl bike start follow us comment eve teas enter shop escap rush away
frequent inappropri behavior encount kirti nagar metro station happen almost everyday
residenti area guy harras lil gal provok
friend travel metro gener compart sit ladi seat middl age man came stood next kept stare make feel uncomfort
person rub bu
sang stuff like quot tune mari ankhiyan dil mei baj ghantiyan tang tang tangquot
want buy chip kmd went kmd alway see group boy stand stare colleg girl stand pan shop
physic verbal abus madanpura near mumbai central station
bu reach anand rao circl morn around 8 40 attempt wake sleep last singl upper seat driver grab breast woke shock realis happen inform reach last stop collect bag amp left bu still state sleep amp shock
guy comment stare girl
man start stalk
night mobil stolen
gone along girl friend watch direct 3d movi pvr boy give 3d glass us touch friend inappropri chest pvr juhu
friend took man hous man offer teach us homework reach friend left man start remov cloth advanc toward lucki escap rape scream run away
teas boy
2006 morn go colleg dilsukhnagar hyd got bu bu station walk forward suddenli men come front two walk side realli close pinch breast went away peopl around saw look dint anyth shock went sat quietli entir could talk anyon dint report polic could see face person pinch similar incid frequent malakpet yashoda hospit bu stop fli
way home even lectur notic two men follow lucki escap friend join
eve teas men women group charni rd station
harass bu
poor street lightn
alon sit near marin drive 2 boy came besid start comment abruptli
drunk guy comment take pictur
board train churchgat peopl pass comment stalk
realli bad
follow group drunk boy way home
ask stupid question man travel bu comment ask number deni final get station felt comfort
drunk man start whistl friend made nois follow group men come shout argu man
station person stare time came near whistl pass took previou posit
incid happen earli morn go school gang boy came toward said hot want take bed
girl call brenda 6 year old abnorm man alway rape
man follow platform touch thrice indec
man beat wife live togeth four year got hospitalis refus sex
share auto morn uncl sit next start touch got embarrass kept bag bag notic thought unintent embarrass
man tri touch privat area push person away ran
forc kiss
teacher beat student
near lake area
lgbt harass male
come back home colleg man stare start wink ignor move ahead
follow senior colleg
comment bu stand even time mostli
follow whistl cross road come tuition
ladi lawyer wait auto rickshaw india gate round teas abus filthi languag goer
friend engag fight usual comment bad word way school home
girl return offic midnight bad boy follow teas
walk friend men spank furiou asham
two car stolen residenti area
fate everi girl dare walk alon road constant uninvit stare quotaccidentalquot touch privat part buse busi place exposur peni even transport buse
girl rape
walk street 2 guy bike stop front scare laugh abus went away
liter less block hous 7pm walk slightli slower normal attract coupl look young guy came behind didnt see come slap grab butt sprint away opposit direct jar especi cuz 7pm close hous blink eye mainli lower incom men close time shiver
happen year 2010 girl 35 year put birth girl child father child left child depend brother 5 year left child uncl though marri fell inlov littl niec wife absenc rape girl wife caught
incid took place near dilshad garden metro station road near station afternoon walk road guy bu comment whistl
teas boy rough word way shop store
walk 4 men gather around kept stare whistl noon around help polic man came 4 men went away polic man walk even interven neither encourag report spoke
survey carri safec red dot foundat along safec audit street market mumbai
incid took place even peopl bike snatch chain young ladi
street light
misbehav
near churchgat station walk realis guy follow also masturb sicken although 0830 0900 sunday street dark entir road mostli offic build showroom peopl street sinc dark area outsid churchgat station go toward marin line
even sit bu stop guy came comment
acid attack
comment catcal whistl
ladi lawyer wait auto rickshaw india haat round teas abus filthi languag goon
ladi pass group men start talk tell quotyou boast loadsquot quotyou carri neighboursquot
chain purs snatch
pedicab cyclist insert impun convers companion happen least twice alreadi
catcal sexual invit guy usual whistl go market read even free time stroll around man older father ever see talk sex
comment bodi
misbehav
man tri touch girl bu visibl uncomfort
go home school suddenli man told like school girl alway preeti
guy take pictur friend talk
foreign happen visit carter road bandra day back found stretch carter road absolut dark found group men sit pass filthi remark women
way town vehicl saw woman run nake outsid men realli anger start insult
group boy follow
man follow whistl
travel bu starnger touch bodi unnecessarili
come back school even two guy came knive tell come lucki close enough flew
walk young man grope crowd disappear
two boy came bike snatch chain go market
happen friend around 3 month back even met man inquir bu rout later ask phone number use call constantli despit disinterest girl shout stop call
cycl kawangwar unfamiliar woma whisper kiss air go even fellowship
 24% |#################                                                       |
told boast shapel
get crowd bu man grope chest
come auto rickshaw two guy hide face hand kerchief helmet snatch chain sped could realis situat neck numb second scratch also
harass
stand queue person behind poke behind
around 4 morn go colleg follow boy follow time twice well
touch grope
even observ group boy call name decent one tri misbehav
stare ogl touch indec exposur
incid took place even near prabhu dyala public school shalimar bagh
know girl harres harres class mate use call differ name say vulgur harres let concentr class
north east girl sexual assault
bu group man touch
way metro station man bike pass obscen comment ask come sit bike
comment afternoon
follow offic
walk road alon boy follow comment whistl afraid even walk road start walk fast reach home share mother
street friend bunch guy comment even took pictur
peopl pass bad comment girl dress style
group boy threw water balloon butt
biker touch chest walk pass along bike
incid happen friend mine met man near street live would keep look kept ignor tri talk becom friendli insist talk phone share phone number start send obscen sexual messag call guy shout stop bother friend
harass happen bu
saw chain snatch afternoon hour girl report polic
around 500pm head toward market group boy pass iwa wear capre comment quotins chote kapd ni milequot
go back home suddenli guy came ask address suddenli start make indec exposur
second visit mumbai along 4 friend sit marin drive group 3 guy sit next us left marin drive notic guy follow cab car like 10 min disappear wasnt enough 5 minut notic anoth car kept follow us till sea link realli make think would someon follow freak sake attent disturb
way station rickshaw old man forc sit middl ladi sit end man tri feel bra turn around gave glare look look way noth happen alreadi hurri let go
boy use stalk everyday use pass cheap comment way tuition
went park pick cousin girl walk pass jean trouser greatest dismay park boy comment next thing saw anoth hit touch buttock bad
walk even time poor light alon guess two guy comment afraid even look
karanja stage girl pass boy start whistl
even 7 pm
bu station town select cloth buy person sell cloth start tell much would cost night ignor walk away
two boy bike snatch rush old ladi chain incid took place night noida sector15 metro station
day ago friend go kashmer gate metro station hold tab hand drug addict came tri snatch refus cut hand
sit rickshaw saw two guy pass lewd comment girl even though threaten report polic happen even
comment fashion
guy propos said use stalk everyday eventu chang tuition time
girl age 16 rape sexual organ cut happen parent sent close shop get good along way terribl incid occur
quotwhen eat thingsquot man ask ladi
rape father year 2012 home live trauma
metro station man crowd tri touch bodi part gener compart
take pictur mine near taj mahal
nine thirti sunday even month may 2011 road mildli lit orang neon light busi walk road near challeng build entranc hand full listen mp3 sudden felt hand touch right hip blood froze fear biker also gut stare lust wish snatch face hurriedli sped wasnt abl note bike reg number
man kept stare breast bu stop
intoler comment
travel even continu comment stare upon
incid toom place gtb metro station gate 2 even group deck girl captur guy guy pass comment
guy follow sister return colleg
bridg desert major ghe beggar drug addict bridg manuy time alcohol bottl cigarett aar seen lie around drug addict brlow east land bridg thri behav wildli mani time chase women
went morn walk saw man approach masturb invit say quotaaja rajaquot come come
molest near store
azadpur area expect discomfort especi girl girl stand go seriou torcher man sell veget touch privat part invit
happen even crowd road two peopl bike shout loudli near friend
navig everyday challeng indian citi live coimbator mumbai varanasi bangalor morn everi act gear toward navig citi go come home late today wear transport use public transport realli need stay late everi thought process guess rel old figur way avoid unsaf time place truth alway fear get rape
school even reach railway line mate lot men start whistl call us ignor like ladi pass closs
rickshaw suddenli two guy start follow bike start whistl pass inappropri comment
chain snatch comment ogl
went shop get good boy teas comment even took pictur scare ran home share parent
go newroad sit opposit side seat whenev driver took break conductor micro intent touch breast pretend happen break
foreign woman studi gokulam mysor 3pm afternoon 2 men drove hit hard breast motorbik drove away laugh say quoti want fuck youquot
happen school hastsal villag uncl approach biscuit took start touch inappropri
pictur also taken morn even
walk work pass railway line man follow stop also stop thought want steal decid put phone insid bag increas walk pace
night usual suffer
neighbour left child hous even someon got hous rape
bu
touch
happen friend get micro conductor tri touch sever time make excus move back
three year ago class mate mine sent school colour shoe go home rather wait near bush wait school close unfortun farmer harass sexual rape end
friend follow group boy tri pull
around 6 pm even return home saw man beat girl snatch neck chain
get rickshaw driver stroke leg
take pictur way knew focus leg
smoke face
nephew 11 year rape mum hous help 23 year mum gone work
pass road saw group boy touch girl butt even want rape girl resist
happen friend even return school went studi 8pm guy ask date refus sexual abus two friend
took place vasai railway station platform board train
ogl comment near post offic area sector 10
guy stalk home bu stop week embarrass littl countri women safeti
walk friend guy ride cycl pass inappropri comment look ignor
pass darajani men usual follow tell stop abus beat
feel safe station safe atmospher gener
room even heard knock door open door visitor long time friend enter room start kiss romanc undress ask go repliedquot want sex youquot
bad touch grope
guy stare absolut ladi goe past front back
peopl follow home also pelt stone comment continu
hi sheela colleg student alway get piss music compos lyricist target specif name glad harass map come atleast vent anger frustrat hope action taken releas movi teesmarkhan target wherev go quit use fact peopl know start sing song quotsheela sheela ki jawaniquot surpris shock know road side romeo start sing song went outsid build get rickshaw ignor first happen quit often coincid harass fed name
unknown person touch privat part bodi
even time guy comment
work touch grope stare thing face almost regular basi
tell us wait us finish primari educ faster
stalk pervert boy till inorbit goregaon
auto driver kept stare ask look ahead ask want get farther road left auto immedi
pass small shop boy know make fun flatter lot like alway ask kiss hug
man take pictur way colleg
lewd comment pass girl wore short dress go back place night call societi question dress way
guy constantli stare afternoon hour
woman molest thane railway station slap molest guy pin hand caus injuri wrist cri help step forward help instead men surround woman demand let go molest men probabl work gang come togeth defens get caught
wid friend walk suddenli saw man masturb got scare
come back offic friend took metro hauz kha chattarpur deboard tri look auto meanwhil group 45 guy probabl heavili drunk pass lot comment start stalk whistl
peopl comment girl special dress
guy text whatsapp till date differ number
travel toward koteshwar nagar two boy approach said danger wear gold walk alon best remov keep bag realiz trick
feel safe till date sometim got unnecessari call unknown peopl avoid
stalk illicit comment pass local boy
man tri molest physic touch
grade six english teacher use touch back sometim use pull bra could speak anyth neither friend
wait metro arriv guy behav badli make uncomfort even
chain snatch
travel somebodi stare follow
 25% |##################                                                      |
wit incid chain snatch afternoon hour ladi later went polic station file report
travel station colleg 155 bu man stare constantli stare back could shout got bu
continu vulgar comment pass group guy even hour
chat befriend differ peopl facebook made guy send ponograph pictur persuad send mine
need protect senior citizen
7 year old shopkeep remov cloth start touch everywher insert finger privat part senior citizen also tri anoth girl fail
sexual harass bu agra udaipur
guy star well comment
man bike follow take pictur phone
incid took place near khalsa colleg afternoon usual happen travel bu colleg hostel vice versa
mother chain snatch market
sometim feel unsaf travel cst sewri ladi coach fight got physic start harass
girl put insid car peopl save peopl
travel bu man tri touch bodi uncomfort drop bu
boy alway call girl refus go repli satrt follow
way school met boy forc remov shirt start touch
happen 11 pm walk stalk man quickli return pg
guy bike tri grope breast girl sit next rickshaw
park near colleg friend pond boy use vulgar languag friend mine
colleagu meet walkov andheri east toward talao pali late r minor girl touch pass lewd remark peopl girl r accompani wid male counterpart watch incid frm distanc use foul languag bad word know night scene happen throughout
go home man cooment
snatch gold chain mother neck group guy bike happen even
catcal bu stop
stalk 3 men around 7 pm return tuition
shock incid report pink citi eve teas common happen girl almost everi well let share 1 incid friend studi 1 prestigi girl colleg jaipur stay pay guest cscheme jaipur two friend went market get someth suddenli 1 car came think 45 boy car tri drag friend car grope wrong place luckili friend save shout dread still cannt forget incid 1 case kind incid keep happen
follow old man everi school hous realli scare scare tell anyon everyon blame girl
dtc buse full eveteas comm enter ogler
area realli unsaf night
comment
sexual invit touch grope ogl
return home school saw number boy teas school girl find girl scare realiz good touch other mental
indec gestur juinagar railway station
incid took place ashok vihar near govern school go friend hous around 2pm group govern school boy pass nasti comment follow
bu stop chabhil face whistl peopl wait pass area walk way
proper street light area therefor becom unsaf girl special night
walk small littl dark area boy stare catcal afraid
friend ice cream pass street group boy comment quot give ice cream us tooquot whistl us
call boy go join school uniform didnt stop cale quotwhat cuti mind join usquot
man follow station hous
got molest gym trainer
aunti know chain ear snatch near sunday market
walk street chandni chowk afternoon two men bump pinch breast wer make weird facial express
catcal whistl
pass comment whistl
follow 2 men bike till locat
teenag girl muzafargarh district south punjab rape strangul death rel local media report two sister play alon hous father away work brotherinlaw brother girl came ask accompani nearbi hous girl rape later strangul rope accus escap scene along infant girl polic recov infant girl arrest accus regist case famili member girl neighbor found bodi rape victim
feel scare go bhera enclav chowk alon night
girl teas gang boy public vehicl peopl bu tri save girl guy threaten peopl say interfer complain anyon kill everyon nobodi took action
wait bu afternoon
walk friend toward olymp stage ayani men work build stop begun stare ladi infront us man catcal loudli notic man said quot ye hold childquot turn look man said quotmadam undafanya watu wasifany kazi hapaquot mean quot madam make peopl work herequot
experi travel train night hour reserv compart boy get station show facial express nowday scare travel train
stalk two boy follow hour tri make convers refus talk got hold luckili peopl around
follow riksha driver seem drunk
indec exposur
comment
micro bu pack stand gave bag old man approxim 6065 watch bag saw old man touch girl sensit part sit next
walk street
street boy comment bad word tri touch
spite unknown person click pictur without permiss afternoon
misbehav
night three guy assum involv busi prostitut start misbehav call polic latter apologis
travel bu peddar road st xavier colleg sit ladi seat crowd bu man push onto much could feel sexual organ arm
robber enter hous rob rs 40 interrupt
girl presum mental unstabl actual time fine fate went farm workwhil farm armbush guy quarter brutal rape got pregnant till today boy found
walk street guy stand lean gate home whistl call whole time cross home take road daili except sunday go music class went month final quit music class 13
sent shope buy food way certain boy start call quotksii ksiiquot dint look back went away
happen someon know come home offic group boy teas follow tri touch fought back incid
survey carri safec red dot foundat along safec audit street market mumbai
incid happen coupl walk lane girl husband grope guy stand nearbi
happen afternoon erickshaw uncomfort peopl around oblivi
stare sing song purpos follow distanc
minor tribal girl rape men dump
ogl catcal
physic abus public vehicl touch front part bodi immedi threw hand seem drunk
friend amp sister walk road even get fresh air suddenli two bike pass away hit hard ass friend sister person sit behind bike
follow home
poor street lightingeven 8 pm
feel unsaf due poor light
incid took place rajiv chowk metro station 24th dec 2014 even guy pinch
harass even
cross road boy motorbik show ugli facial express time give fli kiss time wink realli disturb
man dress militari uniform start follow panjim bu stand till patto sbi atm pass comment facial express weird scari
stand bu stop unknown boy tri take pictur
walk get dark time inner lane men keep comment even find way touch
time deepawali fren return home play deusi bhailo way back guy rape becoz poor street light polic bit secur
guy pass verryyi cheap bad comment
peopl comment tri touch
boy start call pass road
comment
friend shop guy take friend pictur saw shout pretend anyth
ask come greet hous tri take advantag
comment ogl
whenev move outsid hous spot guy stand near corner street pass comment pass know say street bugger
man age 40 follow daili tuition class till home incid took place 3 year ago 11th standard later start show expos privat part inform father polic complaint lodg bike number note trace address found
even time return offic boy comment dress
survey carri safec red dot foundat along safec audit street market mumbai
neighbor would follow whenev went toilet complex kept ignor enter women toilet start forc push ran back home could never tell parent would send back villag
call girl vulgar languag facial express
came natur basket suddenli guy saysquot chalo aapko ghar le jata hoonquot
tri touch differ bodi part teas public area
group girl group boy teas us use slant languag
harass
realli bad
enter insid colleg class mate whistl
two girl follow guy ride bike paa cheesi comment follow way home
group abus kid woman live footpath control male beg night live sleep footpath year also wonder villag group thrown mean street mumbai
thief came cite steal night particular room ladi heard nois neighbour room start shout thiev came room kidnap brought next morn rape
teacher beat pupil
 26% |##################                                                      |
alway comment travel night friend
even walk comment quotchamak challoquot
comment follow
misbehav
man start whistl comment friend walk street
facial express shown also took pictur
first time two boy motorcycl rode past pass close say babe look hot today near banjara hill rd 1thi often happen walk road hyderabad even
visit aunt place wadala boy area harass form kiss action whistl make feel scare extrem unsaf go
chain snatch
treat badli
sexual invit
incid occur school return school guy bike start follow way build start happen everyday till complain parent
survey carri safec red dot foundat along safeti audit street market mumbai
nonsens guy follow tri follow
even friend earn snatch also threatn later went polic station vain
night outsid grant road station
boy gairidhara area whistl follow tri block way feel scare walk area think place alic restaur chines embassi unsaf girl walk even need street light n polic secur place
boy street tri touch friend alon
gone templ sister festiv realli crowd man touch breast inappropri much older disgust
incid happen teacher name ms minakshi teacher school rape later blackmail demand rs50 000
guy stand next sexual excit happen travel bu afternoon
walk friend guy start teas us happen afternoon
stand line grope behind ipu canteen
actual use come back offic everyday
hire cab driver tri touch
follow metro station guy bike pretend phone
man ogl girl touch privat part attent disturb sight
man touch upper part
father treat badli nativ place beat everytim mother brother call polic stop
incid happen broad daylight te assault famili acquaint place exchang book littl know intent back offer glass water rememb shock speak went open cloth suffic hunger sex still wonder happen moment could shout resist happen rememb pain never experienc thing like went shock felt pain inflict tear come could shout may water could use drug could innoc lack exposur anim 16 anyth defend regret
bu
return home tution class two boy came bike hit chest realli hard shout drove away sinc none around walk fast came afraid met elderli woman ask help later boy ran away
street sexual harass walk street saw girl got harass boyfriend sexual firstli girl anoth boy boyfriend saw harass sexual got fight later taken polic offic question
go buy cd peopl ask home tri make afraid walk cd shop guy start follow hide took narrow path escap
walk street night sudden group boy came front struggl harass help came save main time
walk alon street balkumari kharibot group boy sit near start sing song whistl
ride scooter back offic around 8 30 9 pm right breast grab guy motorbik opposit direct hundr meter entranc satyam theatr chennai point lane connect mount road white road
project manag harass repeatedli sexual invit immodest talk
lone road
stand bu stop two boy came start teas put racist remark
group boy pass bike pass comment indec exposur happen even
boss facebook friend send vulgar messag comment unnecessari thing
teas girl pass comment
two guy came stood near start pass cheap comment like free ho naam bataado took place 3 month back even around 67 pm
girlfriend went night buy bread indomi came back look sad inquir told guy harass went caution guy threaten kindli help us put stop frequent area
wait metro guy start comment friend made uncomfort
hardli light road almost street light broken often drunken guy walk road unsaf
7 year old daughter goe school school bu current indian blockad fuel crisi school hire differ vehicl differ place duaghter insist want go school new bu ask told touch privat part new bu conductor went complain princip school safe send daughter school bu frustrat
head home realis two men follow turn back suddenli turn back pretend werent follow
narrow bridg parel station heavi crowd tri climb provid merri ground stalker eve teaser
got comment travel bu
comment disappear
guy ride bike stop near start comment ask lift
eve teas
harass dtc bu
harass
travel bu morn boy comment sang song
cousin sister sexual abus father drunk face eve teas local dahihandi festiv
stand normal grope back
friend came home notic around tri forc kiss touch inappropri
teas area goon night whistl shout name
beed touch
cat call everytim women pass acceler bike scream
go old delhi railway station unfortun went gate 5 instead gate 3 thought keep walk guy pass offend comment
happen even
travel goa school trip train guy stand near door show privat part
flatmat come back amiti univers ola driver switch phone start navig differ rout tri take malici rout jump cab
friend harass whistl man
stand bu stop wait bu boy stand near second later start stare start comment meanwhil second later bu arriv went bu
guy follow us around khan market shop park lot tell polic offic
guy touch becam freez
travel train delhi bhopal guy sit upper berth gave indec look whistl make face
night friend sleep man broke hous tri rape tri fight back man end stab rape
walk mall friend man sit bike talk someon pretend talk someon kept stare inappropri disgust feel
boy tri touch
get scare take skywalk crowd peopl avoid climb stair drug addict drunk seen linger sky walk
comment touch grope
physic touch way back home bu
wit incid stare friend afternoon peopl continu stare leg wear short
touch grope
man would stalk market hous even insist friend way older scare mostli happen would return wednesday market
road side follow group guy teas comment
comment pictur taken night
touch privat part danc partner
harass
efteas
makina stage two high school girl walk school home boy run toward final approach start talk want insist touch hand talk walk faster gave
catcal whistl
lewd comment vashi bu depot
misbehav around 4 30
friend mine boy friend invit restaur met talk sever thing suddenli said let go went ask said show u let go took insid guest hous room part restaur open cloth forc start sex felt bad went back home much trauma month couldnot share anyon month gave number friend say go well sex tri defam
walk get dark time inner lane men keep comment even find way touch
sexual invit
group girl walk school pass group men seat road side start whistl primari school girl
happen road bu stop name gulabi bagh lakshmi bai colleg ashok vihar morn roadsid call sexual invit quotvulgar activitiesquot
student feel safe station
vika puri man bike comment someth amp say love amp
wait get train indian man walk toward pass without make eye contact pinch hard thigh carri walk without look back shock got train
sent shop two boy whistl turn back told she hot
realli bad
friend touch privat part apart mani time seen man stand near ladi compart train leav misbehav sit near window run catch train
friend went girgaon chowpatti walk group boy follow initi thought coincid soon realis follow us decid chang rout
man came invit sex
go back place boy start follow scare lot
attack 4 member gang bmtc bu 9 00am morn tri take laptop luckili got back pull strongli took newli bought mobil safe bangalor careful help
man touch ladi breast cross drainag ladi shout men near click ask quotwhi shout knowquot ladi continu journey embarrass
 28% |####################                                                    |
went triniti boy cheap
sexual invit car pitampura near vip
follow zakir hussain colleg metro station new delhi happen 26th septemb 2013 afternoon
bike follow everi time go nearbi market pass comment
live quarter boy call eyong came harass almost everyday want sexual assault move street met tri undress shout neighbour cameout help
someon whistl
conduct survey worli naka feel unsaf access area congest hous narrow lane dark time men street comment girl feet might someth would even come know
push bu two school boy kept misbehav
diwali 1 uncl sexual harass friend daughter sit pandal made girl sit next touch love
go aunt hous invit function way back home teas boy ran back home
even touch gave bad comment go market
harass home lie tell made angri never want take law hand left everyth god also pray 8th januari 2015
comment metro station uttam nagar dwarka
neighbour tri kiss act though normal
lone road
happen visit male friend hous church friend know togeth start touch becam persist tri grip wall fought until given certain wound lukili escap
roadsid peopl comment other
friend take tution tutor start touch inappropri way
dark place subway quiet night drunk peopl make feel unsaf
men bu grope though happen 2012 still traumatis incid
travel best bu got seat right behind driver driver kept turn look although cover wear scarf got stop came gave bad word left
man around 40 follow daili tution centr till home incid took place three year ago ws std 11 later start expos privat part inform father polic complaint lodg note bike number address trace
accord inform gather littl girl 5 year sexual harass neighbour consid uncl gift use give child
happen rajiv chowk metro station wait metro guy start behav illmann way
poor lightin ameya park
mobil rob becom trend local train lot nerul station
metro even guy comment inappropri girl
age 18 yearstyp harassmentstalk leh market near yebhi showroom time catcal classroom school leh time comment zesma lay look beauti time sexual invit ye even
friend narayanh museum friend queue ticket sit nearbi policeman came said cant sit touch bodi later pretend intent
two fren walk way market sub way guy show us privat part disturb enough gut stand next time also thing stop use rout
man tri touch inappropri road look said didnt react went away
forc slep man infect hiv aid
walk school met boy way start touch ran away
return morn walk man follow till turn around confront
go colleg old man dad age whistl
sit friend two men comment talk us
stalk pass comment
follow boy school
gang goon loot outsid raniganj station
harass
walk street group boy may age 20 come opposit pass guy intent hit breast
eve teas train slap
realli bad
boy stare continu make uncomfort
catcal comment bad facial express
walk road go school man follow came near touch ran go road
go group friend two boy bike came bang friend bum
femal friend hang around durbar squar polic armi look men take pictur secretli made us feel unsecur ran away place
teas boy return home school
group men follow jungl go fetch wood grope tore cloth tell anyon
peopl sit train whistl speak someth bad
yahn par kuch ladk khde rhete hae ladkiyo ko dekh kar heran kart hae
touch inappropri way local indira nagar
comment
misbehav
inappropri touch
crawford market 3 girl go market man dslr camera around neck figur take pictur incid workshop enrich session told us men take pictur post forum discuss pictur saw man camera taken pictur upper lower bodi
man stare smile near hous tri cross path come close pass way work
misbehaviour
forc kidnap rape gun point 2 men move car near dp r k puramafternoon 2 amp 3 pm
sunday night end year parti three us return night club sudden heard guy make funni sound behind us call us continu result us rape molest
friend swim men take pictur us got
boy comment cloth
comment whistl
afternoon constantli stare caus discomfort
saw men take ohoto women pass
due poor light boy start comment look
friend went guy friend parti night tri forc
catcal
way colleg go platform catch next train someon grope butt turn around rush mani quotinnocentquot express left place overfoot bridg kurla station
attend funer ceremoni night wa music men came nowher attack us stole speaker rape girl
man kept push metro
tain journey touch person sit front seat
afternoon 1 pm
even around 4pm shop friend guy cross us whistl loudli
mini bu man tri touch unusu activ
guy tution center keep follow place scare tell parent stop go class
even deni park place guy return sport complex window smash
catcal touch grope even
group guy eat popcorn start throw also pass cheap comment
around 7 oo pm even go back home tuition two boy smoke drink side road pass side immedi start comment start touch came two boy help start scream
age 15 yearstyp harassmentstaringoglingfaci express changspa leh time
23 men catcal whistl pass way home class
walk along jog track end point man step behind bush masturb grin wear blue jean white tshirt look 20
colleg boy came start taunt use bad word
near coach center boy gang pass comment even go class center 40 foota road near kali mata mandir
boy use harass friend everyday front complex morn around 3 month
person know like 7th std tutor call place regard clear doubt goe tutor place take room tri touch wrong place guy succeed though live koperkhairn way thane
night usual face problem ignor
night walk dad wit ladi chain snatch
catcal 2 men societi area
guy bu purpos stick privat part girl common buse mumbai
around month go along friend place even wherein 3 boy look start sing song friend shout happen even
peopl stare rikshwawala came stood near peopl start make wierd comment wait friend rech metro station
way school train boy came ladi coach pull chain ladi
eve teas street earli morn
got follow comment watchman happen residenti area panvel
friend went xma 25th decemb exit got drunk guy went took advantak situat rape hous
incid took place bu stand even 3 guy kashmer gate bu stand drunk behav inappropri
simultan harass catcal follow chain snatch
comment buse
friend walk road meanwhil stranger guy touch thigh stomach badli
 29% |#####################                                                   |
drunkard tri grope girl bridg
bu conductor tri touch breast give ticket crowd bu
guy comment whistl
friend mine grope man ride bicycl
even
idl young men like comment shool girl dress everyday pass alway say walk like qeen
sister touch man bu
whistl
pass toi market men follow go
harass
touch grope
teas someon bu
guy tri intent take advantag crowd bu touch girl good see peopl take notic someth
walk crow road teenag boy came near tri dash
harass school trip male friend
parent went funer left alon boy came hous start touch scream
continu comment stare travel metro even
girl walk makina stage came across group men sit next berber shop call great ignor start insult
teacher beat pupil buttock
comment catcal
teacher beat student
neigbour mine abandon 6 children husband man final gone met anoth childless woman sign marriag certif court
man sit next elbow continu got fed left auto
male touch
grope train bandra dadar first class compart
went market
man fight wife come home drunk wife ran away rape daughter
work intern organis author concern would touch inappropri even told would still stare pass obscen comment final gave tri avoid much possibl
boy end road start whistl walk past
saw ladi afraid two men run behind
go home boy often
friend walk street group boy whistl us uncomfort
friend stand piec dress man notic start make vulgar express
rickshaw anoth ladi person scooter follow us snatch gold chain luckili didnt went hand happen afternoon around march 2012
random guy take pic friendeven 6 30 pm
neighbour sent shop came back gave soda drink later gave 200shill promis anyth long becam fianc
friend stalk random guy kept call messag day complain polic
vulgar comment pass night
friend teas group boy
friend told old man next bu stare quit long time girl opposit wear short touch bodi uncomfort way
touch
man trancul oversleep wife taken hous group arm men rape dump
touch thrice metro station man even walk way
travel car famili worker car repair shop ogl made facial express realli disgust
classmat like follow girl especi go home tell teacher noth
know someon name goe women buy pretend buy rais hand push organon leav themand goe other soweto biashara street
peopl scare due less street light
groupof boy comment togeth make feel insecur
travel afternoon hour guy make weird facial express
touch bodi travel travel
go hospit old man look like mental unstabl touch boob
aw felt horribl helpless took place afternoon
happen dhaula kuan bu stop lane even walk colleg bust stop bunch guy stand start pass comment board bu sinc crowd tri touch
group idl boy sit usalama bridg connect lindi laini saba harass chase even sexual abus girl way school
catcal touch comment chain snatch afternoon
peopl start comment bad word friend
way work man told look hot quotumeivaa leoquot
thursday afternoon class 8 boy tri touch girl
peopl follow made vulgar comment
whilst travel gener compart local train bound dombivli cst man stand behind pinch bottom look back pretend take cellphon happen accid definit case
wit age drunkard woman seduc young boy boy money think that woman seduc touch chick lip boy push walk away
went pick brother school even met brother teacher invit hous promis give lot money saw packet condom pocket took
age 15 yearstyp harassmenttak pictur parti night catcal wa jat look like jat time
male friend model went green room someon touch
comment indec facial express
drunken person harsh comment friend also comment bad word
markit mea ham jate han yhan par mhilao ke sath comment paa kart hae
khoda market friend go mani peopl stare comment us
fix group boy goregaon fast local train alway pass lewd comment women stand cross track platform
walk road boy wistl comment
sister auto auto driver suddenli stop auto came act like check auto actual touch us happen twice us auto driver
guy tri touch privat part friend
girl harass support famili school campu
sexual invit
harass
teacher beat pupil
night guy whistl comment step collect note anoth girl
bunch guy pass bad comment
lewd behaviour
go catch train middl age man tri feel end get piec mind
incid took place afternoon place person proclaim priest took hand hypnot veil read hand weird thing
harass dtc bu
stay night kasaragod hotel otherwis ok staff ask sexual favor
realli bad
gone clinic dental checkup dentist touch neck inappropri start breath heavili sexual excit
ladi wear short dress walk along kibera law court drive men seat along road start shout indec soo embarras
even bu fulli fill worst moment life touch incid happen
maid daughter invit car
hi vicin total mall old airport road rowdi pass comment ogl everi girl pass buy live 200mt hell everi walk offic
realli terribl
catcal comment lot ogl stalk sexual invit khan market group men gather especi around park lot block path stare laugh make sexual explicit comment group men appear 4pm wait cab tri hide cafe also intimid area alley fill men especi 4pm
vashi railway station desert time like central line station bustl peopl henc get littl unsaf travel
stare irrelev unbear unnecessari comment
light amp sound show boy stand behind amp continu push privat part
teacher strike student gather friend place watch movi turn ponograph movi report parent
mother rape unknown men way back market
usual avoid comment guy use follow everyday complain teacher help
got feel sexual harras go school guy tri take photo
unnecessari comment given biker guy comment us cross walk road colleg boy take phone number recept give us bluff call bother us call night time
neighbor use vagu languag shout creat nuisanc foul amp bad bodi languag
touch n comment
touch grope market area got crowd
involuntarili forc touch person
north east india report
girl go home back school saw old man follow
even walk lone street could avoid hurri subject filthi comment weird facial express
friend play ground suddenli boy outsid came societi start teas comment
shop saw shopkeep give sexual invit
peopl follow cab amp pass comment
walk along agakhan walk begger sit said small buttock realli felt uncomfort
 31% |######################                                                  |
husband take care famili provid basic need sinc work depend
incid took place metro afternoon guy stand next metro comment group girl stand nearbi forc friend ask
guy comment happen even
boy local teas sing whistl
attempt rape
comment ogl upon dilshad garden kashmer gate metro station
see road lot boy comment girl
comment ogl
realli bad
come back colleg heard boy whisper amongst laugh look girl pass
walk tuition guy bike touch sped
misbehav
wake road guy gave sexual invit urin
around 1213 year old travel state transport bu parent due rush bu parent seat differ seat away sit differ seat man earli 30 came seat next start gener convers ask class told mention someth mental instabl problem sometim ladi hold babi stand next offer ladi help hold babi ladi hand babi soon took babi hand insert anoth hand tshirt sleev grope breast shock happen also scare even tell parent pretend noth happen rememb stori clearli even almost 12 year
guy call tri snatch chain ran away
boy said bad comment girl walk girl walk guy walk bike
maratha mandir
look froim top buttom comment teas wistl group
riksha puller gang start say obscen thing
metro station man tri touch
saw girl comment boy comment teas call name
go home offic bafal group boy alway pass bad comment
stand platform wait train notic guy stare next stare went around week came ask name colleg name address ignor still ask follow got angri warn bother follow lodg complaint tht never follow
tutori class return home group girl teas give sign sexual invit
bu man kept brush girl
rape husband mood sex moment request afraid unwant pregnanc agravi extent broke fight
street light poorli lit difficult walk street higher chanc get eveteas
group boy blow whistl comment colleg
bu man touch back quit long time nervou abl thing
certain group boy give rude comment give vulgar express happen afternoon
go colleg
two guy click photograph friend public vehicl later rais voic peopl remov next station
grope market
stand bu stop stranger came comment dress sens went ahead stop start stare bad way went away littl tri hide
pinch return colleg bu 29c two stop told exactli word english could slap hand full bag back also 29c also pretti crowd even enough space anyth wish stop
come school morn tuition young man follow ask detail follow long way name felt uncomfort ignor ran away
elbow
man kept stare made weird facial express buy veget wednesday market
catcal young boy follow near marin drive
boy wistl comment bad word
walk makina head salon men sit outsid pub stare wink
harass amiti campu
night b coz work guy near bu stop behav appropri ignor took bu
bu passeng tri misbehav touch intent
use travel mani time 11h bu peak hour morn stand among crowd touch grope behind mani time
walk along oympic road men comment badli certain ladi use abus languag ladi felt inferior
stalk group friend boy republ
go toward moment mall boy start comment us
got messag unknown number repli start send indesc pitur messag eventu block
misbehav
around 0930 pm boy stop car use cheap word catcal tri stop friend way
man tri assualt dark street
usus thing watch peopl gaze road anywher scrutinis sexual organ
nurs student came guy teas say walk alon
last seat man sit besid tri touch first thought mistakenli touch happen repeatedli began feel uncomfort scold man felt asham got bu
boy classroom teas girl say vulgur word alway
walk colleg flat group boy car suddenli stop start comment
12 go wed taxi cram push sit besid driver sinc sit right besid could get much space shift gear start touch right breast naiv ignor later pinch felt realli bad gut speak happen way bandra
poor light area an unsaf night
morn stand bu stop guy start unappropri manner tri touch hand part
travel road alway wit continu comment
push crowd comment group happen even
boy like stop young women women refus follow go
offer free cab inappropri non verbal contact tri stop
travel crow bu time men tri grind intim part bodi
wrongli behav
call name like ldzawa lay hey moon
got grope tourist
peopl stare time never wear arev cloth
group four 2 male 2 femal walk around carter road area middl age man walk grope ass linger take step ahead entir area empti grope could accid friend call start rush away hurriedli post caught demand apolog proof henc could take ahead polic tri involv nearbi polic satisfi apolog
incid occur even harrass plan rape kill certain ladi cought act case report polic
comment
comment mani chain snatch incid happen
femal usual feel unsaf area night due poor street light often electr
teacher beat pupil
girl hostel man use come time use show privat part expos
wait bu peopl comment
peopl pass comment girl pass park
whenev go outsid hostel face kind harass everywher everi time happen everywher outsid colleg hostel
stalk
friend lock public bu harass howev escap
guy whistl stare happen even
tri forc sex
person touch breast shoulder footpath
friend walk back class suddenli man came us start whistl bang friend start scream peopl came help notic man follow us home notic outsid hous day
man came near touch breast insid public bu
group boy non virtual harass girl walk lone small street
cant specifi exact locat incid case happen everyday capit
wait friend bu stop unknown citi elderli man kept star hour
stand bu stop guy came attach bodi back
afternoon man follow take photo without concent especi bub
train man grope peni slap said quotyou touch slap mequotbetween dadar andheri
boy teas peopl react start fight
 32% |#######################                                                 |
friend verbal abus boy
ride scooti travel around would feel earlier mayb ride scooti would eve teas much clearli wrong recent ganesha visarjan notic men stare take excus traffic tri get close friend even normal day find rickshaw driver stare wear skirt rise even wear indian wear still stare lone road car drive behind kept honk even though road empti
salesman ogl tri grope dlf emporioeven 7 30 pm
travel panchpakhadi area thane station tuition ride bicycl man follow bicycl constantli tri speak want know name want know whether ride rout everyday time ignor noth could want rush tuition
happen stadium three girl came infront harass told sex dress sexual gentleman refus
afternoon street walk small brother guy comment dress feel unsaf
classmat tuition class took pictur without knowledg later look phone found pictur mine
salesman kept stare insid store also came store follow
sister go home work truck driver pass dirti comment drunk took uturn took longer rout back home
head home school saw group men sit insid kiosk girl whistl men pass whistl felt uncomfort
play tenni coach continu stare
man invit girl come help fetch water get paid girl went done man lock door coupl hour girl came money
even around 4 pm friend lunch group guy comment quotkanji humar saath bh lunch karl quot lunch us
explain
travel bu crowd guy touch inappropri almost tear
come sub center place guy near mehrauli follow happen thrice place inform parent ask quit job fortun given anoth sub center though inform manag incid later
two year ago went villag sister wed rel touch wherev like time awar sexual harass know say remain silent
bad experi
old person tri touch inappropri place happen bu near juinagar station
even time go shop friend peopl sit insid car comment us bad manner
follow school home man think know live area scare inform parent might stop go school
incid took place morn stand metro yellow line guy comment stare
saw man stare woman make funni nois made uncomfort
stalk
north east girl
return school home subway two three guy follow take photo
guy made indec comment tri grop girl near bal bharti public school
travel bu saket morn man stand besid grope thigh behind disturb
peopl keep stare pass comment night
met new boy friend went place first time start show pnograph movi close eye comfort sinc first date want know hime bett time remov cloth show peni
know girl sister rape naughti boy shesneak hous night good time girlfriend
peopl use bad word pass bad comment girl
mom colleagu ask want learn yoga show interest want learn went learn name teach touch
boy ill treat girl
way themarket met woman age mum ask boyfriend
stare stalk
travel colaba cumballa hill bu man stand next unzip pant flash start fall sit made nois slap conductor forc throw
comment pictur taken even
comment visit market
went visit friend found brother told sit came remov belt want rape succeed
someon stalk
aumti wait someon gurudwara suddenli two peopl came snatch chain happen afternoon
return home group young boy werw whistl comment
catcal comment bad facial express
go zoo touch leg
boy group start teas whistl go school
group boy start follow us
respect elderli rel everytim meet touch back portion massag hug care like
somepeopl dtc bu behav badli
group boy came bike took scaf teas afraid
year 2010 young girl 12 year rape kumba old man 60 year parent girl took matter polic sinc perminet resid kosala know case end
touch grope
touch group men board bu
friend get back home school dda park saw man follow us quickli rush mingl crowd got away
happen girl know crowd bu
peopl car comment
hi sir u ppl done good job im actuali frm visaz state andhra pradesh job purpos got shift hyderabad alon stay hostel meet sister got bu till madhapur guy sat opposit seat frnd wer talk chit chat bu guy suddenli start show sorri unabl describ u clearli conductor back bu frnd unabl complaint conductor felt shi say infront got bu incid taken place nearli 1pm countri chang bad way want share u ppl
harrass regard sexual
touch inappropri place poor street light
guy comment mt morn
8 ovlock mother sent shop buy sugar way saw old man hold school girl touch buttock screem peopl came rescu girl
incid occur return school men stare give awkward express incid realli scare
bunch guy made cheap remark 14 year old girl market
friend propos accept use forc accept much use come home ask meet unnecessari hour
say go school noth instead wast parent money
friend mine sexual harass explain everyth good friend went guy hous guy sexual mode forc girl bed refus guy got well beaten vow never visit boy hous
husband came hous complain affair men ask kind men talk start beat
happen almost everywher delhi
citi increas number eve teas issu month
young ladi 17 year rape 28 year old man ask
cat call grope whisper vulgar stuff ear happen numer time mani time happen broad daylight school grope near station small time horribl place ever think share rickshaw system kandivali east station lokhandwala area safe molest rickshaw variou men time time
boy near hous pursu whenev pass street start sing dirti song say thing like aaja mere paa thing
boy alway follow girl upto home girl realiz follow
bright time two biker snatch chain even though artifici feel unsaf travel citi
go colleg morn felt someon follow turn back saw guy know intent guy certainli made uncomfort
big fan famou nepali pop singer use talk phone long time invit meet restaur went meet talk nice afterward start touch slightli take hand peni afraid could anyth moment kiss infront peopl felt asham went meet onward start hate couldnot share stori anyon
two guy metro station show genit
happen north campu road toward vishwavidalaya metro station indec name call time morn anoth time face indec touch even
blackmail boy
teas take pictur young girl bu two old age man
men say ladi alway readi fuck due dress said certai ladi short dre could bare bend
friend go toilet complex even boy follow month
chain snatch
go buy milk even big man sister came follow guy pat back say wer r u go annoy thing said answer back say person care abt got worri ran home fast
realli bad
govern school 18 man enter toilet girl need safeti
friend harass sanjay camp night 6 month
niec got messag school teacher fb dirti sexual vulgar
survey carri safec red dot foundat along safec audit street market mumbai
comment catcal whistl
walk near bank man tap buttock ask normal
boy use teas girl call stay
realli bad
sometim peopl touch abruptli pass comment
48 year old man attempt rap 2 year old child caught punish law
pass boy start shout suggest
kind lone road return home alon guy came follow street light got dim came closer kept say quotyou seen movi ownex want mequoti got scare went neighbor hous wait outsid hous see actual hous
 33% |########################                                                |
bu
friend return school three boy start follow us pass comment well tri take pictur
electr home sleep terrac boy multiplex behind hous comment
stare
friend way colleg taxi reach destin driver touch wrong manner start flirt ask mobil number
comment
way home micro bu unidentifi boy use filli word point toward disappoint
incid took place rajiv chowk metro station morn around 11am stand line wait metro come two young guy start whistl laugh train came got women coach
seen boy stalk whistl girl
girl travel public vehicl unknown person touch said come deni go later start blackmail say go end rape lastli went
chain purs got snatch
push public vehicl an eve teas
face differ type verbal non verbal sexual harass differ place like public transport templ other
boy tri touch anoth girl privat part
parti fest guy start take undu advantag crowd start touch us
sixteen year old girl rape group four men nubian celebr
tea tea stall men kept ogl pass comment among
walk someon tri touch bad manner
ogl grope sexual invit
abus uncl park pretend tri remov someth neck kiss could stop cri
comment happen everyday colleg
return offic home saw girl short dress guy street start star comment bodi part
saturday morn travel mom meet brother hostel caught local public bu saw young boy touch girl girl feel uncomfort repeatedli touch girl seem bad mood coz incid could much
sister walk chabel chock boy followd smart act like taalk phone n say loudli polish boy ran faster frm
boy whistl friend way home friend insult told would report chief offic
peopl car physic assault ladi
way home met group boy carri girl tri scream help came help girl rape
cloth wearr take shower saw brother use play footbal told dress wear said alreadi put way remov angri shout start fight
go school met man forc girl smoke bhang would sex girl
way shop go home man follow caught start touch privat part tri scream slap
stare
go uncl hous guy follow teas feel sad whole
sister law rape kill brother law near home
group boy alway teas way home
man hold privat part make indec gestur
touch grope dhaulakuan morn time
lone road
go school man come toward tri touch scream ran away
group friend visit water park group boy start follow us pass comment well take photograph
alway face tuition way
happend even vaishali bound metro
chain snatch comment touch grope rape sexual assault
wait friend kingsway came man motorbik ask quotlet goquot time knew sexual invit
group boy harass us
time go dadar stand kandivali station rel go gener compart time man touch bodi horribl like
boy stand near seat girl make indec express bu rout 879
two guy take pictur girl friendss near hansraj colleg
boy click pictur
work woman ignor
boyfriend forc tri kiss vagina escap scream help ran
walk footpath stranger follow tri talk gave respons
boy wistl speak bad word street
friend sexual harras husband inspit famili highli educ 10 year togeth girl divorc 9 year old babi
neighbour tri molest come place excus knew home
walk home gym wear gym cloth group men car start catcal said quoton thousand quot possibl indic much would pay servic impli prostitut
girl walk road bunch guy came along start pass nasti comment
alcohol shop area night 11 group boy resid banganga come harass girl
grope insid metro 4045 year old man
man grope walk broadway road ernakulam broadway narrow road gener crowd easi men come opposit direct touch grope grab
two boy bike snatch girl bag happen afternoon around 12 pm
dinner restuar click pictur return back home suddenli drunk guy start get noisay tri grab hand
stay good societi call regenc estat dombivli first township dombivli east approach road slum around bulb street light stolen henc approach road dark unsaf travel back home
men like call ladi tell readi get marri
incid took place even around 7pm ridg road near welcom metro station boy car show privat part invit come sit car
fetch water railwaylin young men start call tell much money readi give could buy chip french fri
incid took place 21st march 2013 around 6pm two guy bike follow girl pass comment
went market along sister even man around 8090 year age stare inappropri
comment teas unnecessarili return home
gang boy take pictur zoom metro yellow line happen even
boy start follow comment
incid took place dt citi centr gurgaon morn around 10 kitti friend two guy follow tri approach liter escap
group boy comment
teacher beat pupil
sister go home boy comment us
friend travel train bandra harass drug addict drinker ask help readi help ladi compart 45 girl start abus touch
incid remind strong beggar railway station touch tri grab shout gave tight slap front everyon
boy take pictur loo5 pm
4 year girl went miss
go brother meet walk boy call
saturday market everi week lot incid touch grope
happen go friend place
housemaid found sexual abus four year old boy care accord boy repeatedli touch privat part
boy drunk comment bad women tri rape women
man follow friend around templ
group boy wee click pictur friend even hour
everytim pass street creep peopl especi young guy pass comment ogl whistl take pictur
girl invit parti went friend sex parti start friend gave drink never drink drunk extent friend took hous boy rape morn notic die rape 4 abl man
even back function piec guy came upto ask quotwhat charg quot
catcal ogl footov south campu even
even walk 80 ft road around 7 pm even sunday suddenli somebodi back bike hit back head hand time could realiz happen motorbik gone curs step even got back home immedi
man follow see nobodi around touch walk away
travel metro like everi except group boy metro right besid constantli talk whisper look make vulgar sign finger facial express
even area lot drunker around pass comment passer
 35% |#########################                                               |
man call school girl whistl girl ignor went
return school group boy teas dirti facial express even comment
beaten regularli husband especi drunk
nangloi railway station happen
wait friend boy ogl
crowd bu man touch back inappropri
misbehav
boy cross metro station wagon r went whistl pass comment
incid took place afternoon ask oral sex
go home night drunkard teas
walk biker pass comment
stare outsid gate 5 saki naka metro station wait cab happen multipl time
travel back home bu colleg man took seat next first didnt realiz anyth wrong start stick even put hand thigh got took anoth seat
follow comment quotkya boob gainquot quotmaal lag rahi haiquot
night friend travel auto driver took unknown place luckili found help save
number men car follow car took video pictur also stop car call us
saw gang boy teas group school girl although alon still afraid
gang boy tri touch
guy metro stand behind zip open masturb
happen dtc bu travel chattarpur vasant vihar boy contin comment misbehav girl
2 men comment big breast know respond ignor
men walk station sky walk touch walk also touch elbow chest parla station
chain snatch way coach
walk home coach institut car stop besid 2040 sec guy drive car masturb car drove
mom travel guy pass us snatch mom chain report polic await result
train station
man crowdi place tri touch sister bodi part
night outsid grant road station
girl beaten drag hair road
took place even boy ride bike comment tri snatch chain girl walk alon also tri click pictur
man stare half hour train station wait train anoth citi
come back buea step bu two guy walk claim know start ask wrong question exampl quot younger brother mom conditionquot moment mum sick younger brother later ask get home get money sourc 60 000 fr could prepar medcin mum tell anybodi lateron realis luckili thiev
group boy comment everi girl pass
touch grope afternoon
man follow mother return market
wear short group boy pass comment basi wore embarrass felt bad
poor street light unsaf area
friend mine abus kid
agra fort 10 men approach husband photo taken us declin walk away longer interest photo tri take photo cover face turn back even though wore baggi non reveal cloth sunglass constantli harrass
girl stand next appli lipstick boy comment sing cheap song
huda citi centr metro station friend boy start click pictur us
chain snatch
happen morn
whistl demean comment dress wear return parti
travel bu
friend told aunt alway teas follow drunkard pass alley home
aunti chain snatch red light time got riksha traffic move chao
teacher beat pupil
share auto go sec 12 22
man use take care toilet katwekera use tke advantag ladi use toilet found rape 6 year old girl
guy drunk rape girl fren place live relationship liter throw nake window gurl found dead next morn polic file complaint said gurl affair guy well
friend go back home school 2 boy school uniform start follow us comment us boy made fun friend breast boy alway stand outsid girl school creat problem us
friend walk hostel van prison pass whistl made indec comment dress
go shop saw friend corner sorround boy touch want go stop thought would troubl ran
catcal ogl stone throw women common particularli colleg buse lectur even femal turn blind eye pl avoid colleg
test
someon touch inappropri near vasu dev school
walk friend boy follow us
girl sexual behav touch bodi girl cri soo much
bu station sister heard someon whistl us
incid took place 4th octob even district park janakpuri guy car chase girl know metro station hous
stand amp boy continu watch comment also follow till reach metro stneven 5 30 pm
go school group boy teas comment
travel bu alon guy behind start sing song
happen outsid school afternoon car stop front man insid car began expos show privat part
wait train dadar station saw man stare continu stop even gave gold glare
2 women sit bench privat convers disturb secur guard busi stare us intrus request sever time stop explain ogl disturb ignor request took photo made nervou tri scare us delet photo refus threaten hit whole incid degener assault us neighbour misguid guard disproportion backlash stand street harass realiz angri mob confront us tri call polic amp women cell avail basic seem neighbour stand peac firmli street harass greater offenc street harass
group boy follow back commun come back dwarka metro station
go school group boy start teas start laugh walk quickli area avoid
follow coach centr hous everyday guy kept pass cheap comment
matatu karanja stage saw tout friend call ladi sell cloth ignor start make fun
survey carri safec red dot foundat along safec audit street market mumbai
remotewala man basic deal use stuff man use come societi collect use stuff noon time near corridor man pass suddenli start undress actual show peni gestur whether want dadarkar compound tulshiwadi tardeo
fate afternoon head road errand notic man look manner made uncomfort even pass call told usual thing girlfriend shock consid age refus becam constant afraid pass would see
way colleg saw group peopl comment girl pass
comment pass two boy ride bike
subject lewd comment grope men
wait metro platform saw group boy 3 4 sit corner comment upon girl alon girl could even face walk away head low happen even
catcal sexual invit
friend told walk road group peopl comment use differ word
gone khanpur shop 34 boy start follow us walk faster went narrow lane lose caught enter shop saw women wait boy leav women ask us everyth ok that told happen aunti went confront guy made leav
men wink us pass throuh rout
guy argument girl amp abus enter ladi compart time afternoon 22 30
get massag hotel masseus insert finger vagina without consent
age 15 yearstyp harassmentcatcal 2016 hemi villag time
student depart comment indec word
eveteas
cross road bunch rowdi rider deliber swerv came right scare make cat call rush
friend scooter boy follow tri grab forc
group guy follow 34 bike
get inappropri phone call whatsapp messag alway stalk
neighbour come hous weekend half nake littl sister play goe ahead start shout attract attent
invit former classmat home help studi sat biggest chair togeth studi well until start touch thigh move away came closer forc end studi behaviour
catcal whistl morn kalyan railway station
guy bu travel sinc seat empti besid sat purpos push
follow stare
touch comment upon afternoon travel
touch grope travel delhi metro train gener compartmet
friend mine come place near andheri station boy stalk
coffe shop two girl walk start make convers told man follow around take pictur girl walk leav alon
crawford market photograph slyli took pictur breast focuss went confront deni take anyth went ahead check camera found pictur made delet
friend mine invit date littl know invit lodgingl room insid ask boy book room boy smile told hide feel ladi long time want tell boy remov cloth expect ladi refus fold corner frighten boy left
 36% |##########################                                              |
happen time birth right comment
girl sexual abus grope public holi
rape sexual assault even
guy sit behind bu start sing song direct
stalk eve teas touch grope
young man shout whistl girl pass tell beauti
morn guy keep call shout name go colleg refus go wit movi
boy seat stage whistl sister make comment bodi structur
follow guy colleg bu stop usual go
follow
way school often face teas boy
cross sarojni nagar subway night amp drunken man follow amp comment upon
walk even talk phone suddenli guy hit hand walk away group friend
come back colleg boy start take pictur start comment metro station
misbehav
guy purpos grope metro stop
go class boy follow home till class bike say quotgiv number friendquot
girl go road suddenli person came start say lewd thing realli discomfort girl
night extrem poor streest light unsaf cross ground
touch micro bu
sexual invit girl
even boy comment
return class guy said indec word victim
walk infront kathmandu mall street boy show privat part
walk toward church jamhuri estat man stand road side approach ask rose told ask name say anyth strang sped went
sexual abus return home colleg narrow road
victim notic two peopl constantli stare ignor think passersbi nowher two peopl came bike snatch gold chain snatch caus deep cut throat
15 year boy ever march sister find group boy teas comment
10 auto guy tri extra rear view mirror head strateg place view passeng torso ladi travel purpos go fast everi speed breaker glanc mirror sick sick sick
go school unknown boy call even tri follow
guy make differ express
touch grope
follw school 5 6
friend abus boyfriend even slap market told dress conserv
boy way home make wire facial express whistl stare
stare touch grope ogl
wit incid man harass girl refus go hous
get whistl main market
walk road even group boy bike start call name sing
went gateway india crowd usual felt neg vibe crowd back man pass push shoulder thought could avoid done purpos
comment
catcal ogl
subject whistl guy even
survey carri safec red dot foundat along safec audit street market mumbai
night near railway station group boy comment tri touch
harass
incid took place 10th sept 2013 afternoon wait bu stand boy came pass comment two girl stand
incid took place april 2013 even went nsp netaji subhash palac friend snack two boy came comment friend alreadi late hurri go back home react
harass
touch
pass certain path night saw man touch woman buttock woman like start call man name
comment grope hoot
happen near shopper stop even
late night around 11 30 middl age guy gave lift ride push toward tri hold leg thank escap central mall indor junction station dark alli bridg
girl rape road night street light
pictur also taken afternoon
crow person touch back pretend accid
rel place aunt gone alon home uncl call room told sit close start touch person bodi part tore inner shout start scream got run away threaten told anyon hurt badli
physic abus grab boob walk emot weak sever weak
group guy chase ogl
man took pictur public transport
way home met four men know circl rape
pass area full boy came spank buttock without knowledg complain start insult
gaze boy
feel scare use skywalk way men shove
follow man bike
touch cantt area even
someon took pictur without permiss
teacher beat student
went mcd sector 14 mani peopl vicin drunk start stare teas quit time
face difficulti mani time pass group boy teas give variou name well whistl
water park someon touch breast
man misbehav
http news outlookindia com item aspxartid785656
chain snatch comment area night
asgarali compound gaodevi behind gate near landlord gate proper light drug addict gambl take place
two boy take pictur girl also comment girl got embarrass
guy tri come closer touch travel metro even hour
travel bu boy intent fall tri touch
wall climb adventur island guy ask quot what bra sizequot
happen j block market
go rel place auto guy start harrass
guy tri click pictur stare come kurla station
eve neighbour room girl rape happen girl visit boy claim friend finish discuss girl want leav boy pull threw bed start beat faint rape notic girl regain conscious shout start cri
sexual assault night drink friend acquaint tri forc get cloth
walk girl said hey babi need want sex exchang money
group boy pass comment girl metro
may ask go scare aso ran away
metro
drunk man tri kiss girl pass
catcal comment touch other buse
daughter rape youth sit along river pretend clean river
riksha guy misbehav pull dupatta
girl pass guy comment cheapli vulgar way even hour
group boy stare badli pass say unwant word insid bu
felt harrass men
friend tri shop market take bend found group boy made advanc whistl even hit bottuck got mad advis passebi let go
man flash public start catcal ignor
 37% |###########################                                             |
gone morn walk wear top text written boy pass front read text loud start laugh happen 67 month back
even 5 30 pm
guy pull scarf bike
whistl near wadala
metro civil line guy click pictur check phone mani vulgar pictur gave phone secur guard take strict action happen 17th august 2013 even
catcal
incid took place februari afternoon vulgar comment pass regard costum inter colleg competit
touch prime hour busi spot andheri station person pass rickshaw simpli touch breast ran away sat disgust dumbfound rickshaw signal couldnt even see face dont even know hate time point
walk boy teas sing song
young go parti famili member public vehicl adult sit next stare us got afraid ran famili member
12 year old girl
phone stolen
touch grope afternoon
guy crowd bu rub femal bodi
incid took plae modinagar way home wherein underw worst situat life pass street two biker came obstruct way nd start pass lame comment beyond imagin awstruck seceond jitter mind rang parent sinc hous far away father note vehicl number bike file complain polic station
walk road group boy thre n say mani thing n bauti sexi feel bad n feel un safe
bad happen night
wait bu two men bike came close start sing lewd song stop took next bu came even though suppos board
survey carri safec red dot foundat along safec audit street market mumbai
happen near old pump hous
survey carri safec red dot foundat along safeti audit street market mumbai
take walk area park even underw filthi comment
last year follow home local nightclub man tri talk coupl outsid hall know could happen
friend mine harass bu
stalk ogl seen resist
man start comment friend pass
ogl sexual invit comment touch stalk
stare near metro station
ye someon call next open ground
group school boy pass intoler comment go back flat happen afternoon near roop nagar market
group guy tri follow throughout r rajpal way
go templ sister public vehicl boy touch hand unnecessarili told sister ignor
follow andcom say quotkya boob hai maal lag rahi haiquot
follow boy
way meet friend 2 men pass nasti comment cross
pass men start whisper
leav banglor run late could get cab driver tri touch thigh took place near vt station sinc hurri could revert
two men call friend went gave money
complaint regard poor street light ulhasnagar station difficult use bridg night
walk inner circl boy use heavi tone make obscen voic
rikshaw puller tri take wrong rout howev panick wen shout way ip univers sector 14 metro station
taxi driver tri feel tell bug shirt
guy tri touch inappropri place
near munirka guy follow one munirka guy car stop near tri get push insid car manag run away
catcal market call slut hindi
sion local railway station complet desert around 2a night take advantag gang men tri kidnap women drag platform toward main road polic help avail point time eye wit tri call polic helplin effort vain toward dadar side bridg
man saw friend underpass masterb front use faul word address
whistl catcal
constant stare pass comment
happen ear toilet complex night way toward tge toilet usual surround group boy stare us month
guy keep masturb teas sight girl kept come back despit complain
go offic felt guy knowingli touch bodi part
women get harrass polic nepalorwhat happen today let tell happen today want get food scooter put signal stop second sudden somebodi hit scooter realli hard bike luckili mani peopl around driver ran away afterward left behind friend tri run also peopl could hold back go charg driver probabl licens never show tri apolog stori want tell happen go polic station even wors thing like happen never happen alreadi nepal hit much harder pleas consid also obvious shock accid constantli polic men polic headquart make fun make look stupid treat like capabl support tri help even talk properli even tri talk nepali english talk male person call tourist guid call accid case even scooter present accid occas address chief polic charg start ask number call sexi imagin kind state mind accid polic take advantag itw talk polic headquart kathmandu happen back room somewher nowher justifi case also happen public peopl present observ nobodi spoke polic treat women nepalhav privileg posit foreign even want imagin nepali sister get treat embassi back polic men afraid behaviour realli accept ignor polic serv public protect peopl
look post tech harass felt abus coerciv categori use ask upload pic begin ask upload normal pic pic short gradual ask strip refus use get angri stop talk excit first like burden told seem concern first eventu resum pic request never threaten forc relationship stake want know whether abus still confus
stalk cooment local boy
group guy stalk friend
pass men sit shop verand said look sexi min skirt
sit next door nepal yatayat man stand stair bu door face bodi part touch hand unpleas ask stand still
walk home young man slap bum pass freez comment anyth els ran
rape took pictur strangl almost death
two men kept stare pass lewd comment start walk away would like bodi
gang biker other scooti round front group girl block exit area
happen nehru market rajouri garden even around 5 friend gone market guy whistl start sing pass also gave creepi look
comment pass classmat
two girl walk night men call went got drag corridor got rape
happen near colleg vocat studi sheikh sarai disgust guy said absurd stuff behav disgust manner said thing express
ride scooter microbu overtook driver conductor blue micro bu said quot new ride scooter slowli ridingshould teach ridequot type comment got sever time
proper street light guy comment
incid took place afternoon dilshad garden market girl walk road two peopl came bike hit back abus badli
friend complain tell stori group men like comment bodi structur say big breast boob behind
travel bu man tri touch put hand shoulder
86 year old mother attack chain snatcher build 3 decemb 2012 scuffl fell fractur thigh bone
friend molest boyfriend
place crowd festiv newar commun call jatra man touch harsli
afternoon friend guy pass us rub us
accord told guy 20 yr ride machin live alway come back home even saw littl girl 12 yr return sent caught littl girlpush insid slap rape
continu rape two uncl contin rape niec absenc child parent
two girl colleg pass out rag new admiss wish
sexual harass happen friend mine still secondari school christma break gang thiev break hous process rape act disvirgin truamatis hate men take rehab lesson
boy continu star
come back offic auto driver intent took auto wrong path dark star continu
guy make filthi comment behind back pass even whistl happen afternoon
happen near entranc moolchand metro station even group four guy stare sing stupid song wink
misbehav
ogl catcal
someon grope behind
episod happen 20th birthday decemb 2013 though found later friend told littl gone club royalti bandra west danc older man foreign came friend felt back move anoth friend becam extrem touchi friend clearli comfort first friend went club bouncer ask help said till push away cant anyth friend shock ask even rape wouldnt anyth bouncer shrug shoulder
friend walk olymp small path man take alcohol sura mbaya roho safi pub shout call us go pass quickli avoid becam louder insist go share alcohol drink call thrice took heal left shout
incid took place near batra cinema even go area boy bike pass rubbish comment girl wear skirt
morn walk feel sexual harass man walk nake show privat bodi part walk side side ignor possibl
vulgar express comment done return coach
sister travel nepal yatayat old man touch upper bodi sister ignor
go market two guy came bike stare cross circl comment
comment catcal touch grope
travel region india sleeper bu man respons check ticket kept open pod door harass got bu rest stop use washroom attempt drag away field nearbi thank famili saw happen rescu
walk home work came across guy start whistl look make sexual facial express
 39% |############################                                            |
closer mumbai kalyan mental challeng 14 year old boy sodomis 4 men shave eyebrow head burn privat part cigarett butt forc unnatur sex parad road later accus beat father dissuad approach polic fourth accus person still abscond toi mumbai edit say sexual assault rampant
go buy veget two men come blink eye shake tongu
man snatch chain train
friend forc sex toilet
board last paper crowd person touch back
comment touch girl bad manner
inappropri comment catcal whistl dtc buse
go home alon without friend travel toward gwarko peopl bu whistl gave facial express
travel bu conductor harass
boy studi 9th standard saw two girl fight caus boyfriend click photograph upload facebook also sexual harass
teacher beat student
complet slum area fill drunkard drug addict amp area lot murder report day bewar go next sea amp could thrown sea tortur help
accid road poor street light rough road
guy travel rickshaw kept look back weirdli repeatedli
bike two boy sstalk friend like 510 minut last went polic men way biker ran away
follow boy school home
dtc buse amp arsd bu stand touch whistl
went husband hous resolv matter fight along brother 15th june 2016 around 7 30pm 4 month marriag came start shout tell brother get lost hous said say brother turn told get hous start abus filthi brother pull brother start push throw hous went stop push brother push harass parent also involv 3 e husband father law mother law start push hous tri pack bag fifthli abus parent pack bag start throw bag hous rain even kick bag hit pick thrown bag outsid rude behavior first time abus lot time verbal abus say mother fucker rand chinal push sever time file fir complaint polic famili safeti hope peopl help give right pusinh husband name harish kumar work netapp father law name maruti rao mother law name indumati look forward help thank nikitha
comment 3 boy school realli bad guy dont want studi harass girl
ladi bu stop felt exploit two men kept comment look
travel bu group guy teas girl well tri touch later pass comment well stop bu go
middl age man put chew gum hair
return colleg two boy make indec nois front face went away
come home church place dark saw person look call ask give money told noth threaten deal ran
16 year old girl badli beaten parent relationship issu
3 guy misbehav yesterday till inform husband run away safeti tourist srinagar mohmadmufti womensafetyat new krishna hotel srinagar guy touch inappropri felt proud pack thng shldnt happen
woman rrescu men want rfape escap
guy sing hindi song like hurt tri teas touch come metro
return home colleg group boy follow train stop way
man randomli appear front ask certain locat show direct later ask em accompani smile say safe
walk road guy teas nick name throw paper etc
car stop pretenc ask direct later boy offer unwant lift deni follow till took refug safe crowd place
noy comment girl pass near metro station
catcal comment touch ogl dtc bu safdurjung enclav hour night
group 5 boy harass girl happen even hour
morn time usual face go colleg
street sexual harass wit molyko buea girl put known hot pant guy smoke behind bar friend drink went girl choic guy mani rape help sinc im girl ran insid bar
go shop met boy know start whistl ignor came grab hand forc go hous refus told oass sinc neighbour think wil let go insist took hous want sex refus start scream help
walk someon tri snatch friend bag motorbik
comment
incid took place 3 month back even around 67pm two guy pass cheap comment
everyday visit place kanhaiya nagar tri nagar boy take pictur girl happen even
travel bu saw girl sit near behind 23 boy stand pass cheap comment girl knowingli ignor need travel daili
touch grope inappropri place
group peopl whistl passerbi annoy call name irrit
mumbai suburban train men get grope men crowd compart regular occur need address overcrowd anyth
pass call refus abus even start follow home
public bu guy tri grope friend femal part even impli stop stop
walk friend chemist shop nightand wear short suddenli two guy start follow us start pass comment
teacher beat student
wait auto two boy leav bike kept whistl stare
eve teas stare comment
old guy whistl comment colleg girl happen morn
6 yr ago wait micro bu man star spoke abus word even follow
sister came home cri lewd comment man pass said want take home dirti thing skip school ten day fear
man scooter stop front said quotnic boobsquot drove shock react till much later
survey along safeti audit
saw guy touch girl bu
aman touch boob intent walk along moi avenu place croudi advantag pice di react stranger
confectioneri shop near home
friend walk behind coloni street man came friend grope waist
pass market group guy give absorb look sinc wear short
girl kidnap rape crimin
wit obscen incid today man entranc palladium build masterb public law broken come section 294b incid report action seem taken secur personnel present
incid took place even old man age around 50 start follow sister dhaula kuan even click mani pictur
way gurgaon today morn follow two boy maroon wagaon r number dl4c aa 6837 cheap piec nutcas werr click pictur make video drive talk women safeti women harass mani level actual fail nation importantli human be
school teacher harass us say love us touch us teach us disturb act institut like school
matern aunt go rickshaw two rogu snatch gold chain behind happen even
girl wink start feel love
two boy bike comment abus word three friend
girl return school alon home street boy follow teas
ankamma templ stolen money
polic man behav inappropri way gay men
girl travel metro guy around start catcal interven around
walk along agakhan walk hold hand friend man normal beg uchumi ask money didnt give told boyfriend die hiv aid
pass near ther bridg boy start call girl dress trouser
walk near metro station guy car follow got rikshaw hit car afternoon
sexual invit fb
ladi wear jewellri went friend hous attend pooja return two boy came bicycl snatch jewellri
person tri touch fulli crowd bu
girl cheat sleep adult want
follow biker also pass comment
happen bu stop
take scooti park peopl coment follow
along friend along ridg area three peopl follow us eve teas
home sit outsid door neighbour call hous cheat sweet
go somewher dda park shorter rout men play card park start whistl comment ran happen afternoon
sister go school boy call tri touch
face verbal violenc virar station two day back
taxi driver harass girl univers buea junction girl complet paid taxi fare girl explan agre 150 fr girl give money driver said fare 200 fr girl left driver came shout
comment ride scooti
loud banter guy
happen travel micro bu boy tri touch
touch
 40% |#############################                                           |
happen friend mine wait open forev 21 saket felt someth touch bodi behind 5 min turn back see middl age man unzip pant
man stand schoolbu build wall north avenu santacruz west masturb expos pretti dark wall bu enough bush hide watchmen 1012 feet away initi unclear owe dark becom evid call watchmen ran away disturb kid live build around also right busi arteri lane link sv road complet lack fear discoveri busi locat also disquiet goe show even seemingli highli unlik tini seclud spot becom opportun
walk alon road man came behind touch shout abus word ran away
friend mummi chain snatchin hui hai
travel bu crowd old man touch hip time left bu
happen sunday afternoon around 6 month back market mom boy constantli follow scare could even tell mom later mom insid shop start beat boy mom look saw beat boy
walk river bank dirti look man follow us show us privat part
walk friend mine saibaba coloni pretti residenti area man bike wear helmet grope ride bike frozen gone swift street dark even note vehicl number disgust
sever girl siloam silanga stay rescu centr secur exploit sexual activ form abus lead earli pregnanc
bu heavili crowd guy rub
friend pull bush dark park
go rel hous follow friend propos accept propos
girl rape last week night stranger come school head home
harass
go yoga class morn guy bike made bad comment invit
follow man
age 16 yearstyp harassmentstalk leh main bazaar 2015 time take pictur leh main bazaar 2015 time comment school leh time
pass comment late even
boy simpli touch figur
touch comment dtc bu route140 even
sister sit outsid door two boy pass start say beauti would love us bed
sunday market nuisanc man tri grope market interven inform polic blame cloth
drunk rickshaw puller comment ogl
realli terribl
age 16 yearstyp harassmentcom school leh time
friend sister walk attend birthday wear dress group kawad pass hoot pass comment
friend mine femal sexual harass workplac old contractor kept give remark look report time delay posh dept compani
school girl touch two old boy saw coridor noth boy big
happen morn
come back school notic drunk man follow kept ignor walk faster alway move around friend
man kept touch girl thigh pretend happen crowd went told stop call polic girl thank
man comment figur wait train made vulgar face gestur
man area caught red hand neighbour tri touch nurseri school girl privat partd claim help girl remov pant urin
return home metro station men start comment
comment chain snatch catcal area even
go station road besid group boy pass comment
saw two boy bike around 20 year age eve teas girl walk road probabl metro station
guy school tri pull dupatta school hour
misbehav
survey carri safec red dot foundat along safeti audit street market mumbai
tri touch butt stare continu
certain girl left school class five sexual abus boy use make fun that decid left school
boy stand near seat girl make indec express like touch etc bu route879
chain snatch
usual cng buse happen head south ex
way gatwekera met two men road kick fell start touch breast
man tri touch insid bu
guy pass comment friend start sing item song pass felt
bu man brush wrong way kept touch crotch shoulder show discomfort clearli take hint got left
pinch comment crowd buse often
sexual invit around 8pm 17 06 2013
go colleg guy start comment near metro station
talk doubl mean comment girl
return home bu man ask name comment quotyou look beauti moti fat girl
afternoon hour bu guy stand front give stupid express
sit bu guy tri touch got irrit shout happen rout 764 bu
catcal
mistreat boy
stare whistl comment
boy comment girl pass touch girl react comment quoti bahan choo logiquot demean cheap behavior
whistl wait bu
travel rickshaw alon suddenli turn unknown direct tri touch
four boy comment badli follow girl 10 min
guy look sing obscen song make indec gestur
inappropri touch
man whistl shop famili
took place even boy stand front school misbehav comment
summer afternoon walk street 2 boy motorbik come toward opposit direct touch lower front part bodi
realli bad
group guy start sing song whistl friend walk back home tuition
girl rape dad
school girl touch man resist ran away
guy comment go class friend move distract
year back go near shop aiyappa societi madhapur hyderabad road dark none street light suddenli guy came bicycl grope
wed ceremoni man tri teas sister follow watch us
unfaith took sweet potato girl date took bottl drink got drunk took alcohol drink took home help refreshen still okay saw opprotun sex admir bodi long time wors thing ever happen xma
teacher call break offic teacher start touch ran away
happen friend man middl age tri touch friend take advantag crowd
head bu stop board vehicl school kenyatta univers suddenli man nowher came spank start laugh walk away mad
harass
stalk stranger return coach
senior school bulli laugh
child work dhaba beaten owner happen afternoon lunch
stori student villag stay citi studi purpos 2 cousin sister elder brother use live rent room brother establish sexual relat sister result pregnanc
face kind stalk mani time come back colleg late even hour sinc even colleg stalker even approach tri talk
two men forc thought drunk start touch nt
survey carri safec red dot foundat along safec audit street market mumbai
colleg trip train ride guy look ugli way guy soldier civilian cloth happen train back goa
even around 530 come bu stand home foot biker came back grab chest ran unfortun could note bike sudden could even realiz happen disgust want yell beat tel public properti
observ privat bodi part
enter subway someon go friend go cross grope butt
run tea stall near lal kuan petrol pump mostli cater men truck driver night around 10 pm three drunk men came stall start pass lewd comment pull dupatta son 8 went call help poor woman peopl blame point call polic corrupt team men
abus man cloth
mostli travel buse train et happen
catcal sexual song
man came close said look hot want touch afraid
sexual abus public vehicl
lot crowd space outsid ticket counter toward thane end mulund west plaform usual male youth part way let deliber brush past tri make way
person pass cheap comment happen bu stand shahdara two year back earli morn
way work everi even see guy smoke indian hemp touch sex dark road side bu stop mind busi faith came across girl violat want help badli beaten purs phone stolen kindli help us map area peopl awar avoid rape harass
harass
catcal 2 men near park
 41% |#############################                                           |
rough road traffic problem
indec exposur masturb indec comment whistl jostl touch other bodi
saw girl wait someon tri call person meantim came three boy teas well tri touch
person touch wrong place travel bu way home
morn happen
teas comment
walk escal got grope man stand behind
young girl defil old man stay noth done
man sit bench listen radio comment said love pass embarrass moment
touch old man microvan
age 19 yearstyp harassmentfaci express leh market time stalk leh market time catcal tuition time leh time
neighbor rm nagar use vagu languag shout creat nuisanc foul amp bad bodi languag
even boyfriend comment quot humara bhi haath pakad loquot
group men two motorcycl follow auto kept come side stare us kept ignor got eager kept circl auto till final pull went ahead
way back work men around darajani grab touch breast
man whistl pass uncomfort
stare
purpos tri come close touch talk crap
go husband offic fellow came ask favour bike show scare said noth went sinc today got site thought share nightmar
boy teas go sister hous scole way
walk way home dark alreadi group boy start flirt scare ran away reach home fast
happen near dk metro station two men ogl
full pack metro person forc come close girl happen even
stand traffic signal front hyderabad citi centr mall good auto stop signal autowala start honk look pucker lip vulgar fashion sign ask would come auto number ap 11 x8604
watchman tri kiss tri mani time hide stair tri touch wrong place
crowd vehicl middl age man lean back even sometim bu space didnt move
man tri follow stare
man come close touch privat bodi part
go teen function wore tight pencil tight top pass toi market male trader start comment neg dress code told love tight bodi long grab real sex tight hot bodi
harass
harass
grasp hand
alon tri catch auto man tri touch improp manner face shout back smartli ignor walk away pretend unawar happen
man touch aunt behind went shop
walk school head home ladi knew bodi shape kept whistl apar reason
stare
group guy make face friend
somebodi hide behind tree call join
yell abus pass decent remark girl touch inappropri place
touch
notic stalk old man way work area anyway danger poorli lit decid ignor walk fast could
walk guy pass remark us
rickshaw driver hurl abus
guy group constantli stare stalk two girl afternoon
touch privat part lift
happen friend harass public place even slap
eve teas catcal lewd languag abus victim respond
mate man way school start wink us never respond
comment strang men near ramja red light got colleg make sexual remark
group 3 guy start whistl walk colleg station
teacher seduc girl beat ass
person stalk sister everi time goe alon
happen gurgaon
saw bandra link road 2 boy take pic girl back
person tri touch privat part stand rush dtc bu soon could respond got bu happen afternoon even
guy comment
walk dinner bike pass 2 person sit whistl cat call
two boy comment sister return home market
head home group guy start harass bu station hold call name like quotmi sizequot quotbabyquot warn leav alon walk away next thing blow insult hurl could noth walk away
took son went versova beach mumbai around 6 30 p last sunday got scare look water start shout group 6 men came near us start convers amongst afraid water kept inch toward us surround us felt uncomfort leav place describ yhr incid friend told safe place go even
report taken dadar street market interviewe feel safe instanc stanc visit place
friend go tution guy bike spank butt ran away
go back home guy sit car saw start follow pass comment
survey carri safec red dot foundat along safec audit street market mumbai
even men tri touch girl girl slap
touch grope mangolpuri
girl bought chip worth ksh 10 boy told eat go relax hous
queue platform man push touch woman badli woman slap time platform 9 thane railway station
incid took place school boy class use teas call unnecessari name
girlfriend group middl age men watch porn public seem drunk wait hall isbt kashmer gate
touch inapprori
follow
travel bu panjim miramar boy willingli fall friend
metro station wait metro guy stand littl far away start whistl look
touch public transport
catcal
man took advantag crowdypath touch anoth ladi
go pakiza buy stuff way person start follow scare start walk fast quickli enter store went somewher hide
teacher beat pupil
eveteas seen report
happen march 2013 two biker snatch chain woman go market
even observ group boy call name decent one tri misbehav
guy use follow girl whenev home guy everyday routin girl scare come hous
guy comment
guy follow way friend home
walk park boy comment appear
aunt chain snatch gone local market buy veget
insuffici light feel unsaf travel even
man misbehav ladi bu stop
snatch chain even
travel public vehicl unnecessarili touch unknown person felt uncomfort call nearbi sit person took anoth seat felt relief
touch grope
alon bu group boy whistl comment even tri touch
swim pool boy tri make situat touch bodi part
men call girl refus start call namesand comment featur
comment catcal even
sister go market two boy start sing obscen song look us happen afternoon
bunch guy kept stare good 15 min thu make extrem uncomfort
went shop 3 guy stand near shop approach first start make facial gestur start whistl take pictur
even visit friend group guy would stop comment
go uncl hous chabahil met person street way start whistl ask start name deni give name time feel unsecur
brother tri rape stab leg ran away
way school saw ladi pass group boy told quotyou ladi come els make pregnant
friend walk group guy start walk behind us start say quot join youquot walk faster comment quot walk fastquot start follow us comment quotwil girlfriendquot refus walk faster still follow us minut stop follow
grope wait line metrogrop dtc buscatcal whistl
survey carri safec red dot foundat along safec audit street market mumbai
travel public vehicl crowd sit seat kept bag lap could feel someth lap remov bag check wrong saw middl age man tri touch privat part shout left vehicl
 43% |###############################                                         |
ill treat bu n road
grope metro station kanhaiya nagar
follow amp tri come closer
woman stand bu touch boy uncomfort manner felt mani time slap boy
comment hoot passer
stalk
take pictur
boy eveteas upon group girl
friend come church met old man call friend help opper phone friend went assist find man want watch porn video phone
bag snatch riksha happen lot
touch grope
middl age man open zip crowd metro teas touch girl genit girl along men shout caught hold man beaten bleed profus apologis hand metro author
entri exit bridg malad station start end near men public toilet alway crowd men stand also path main road kind compound morn even boy men stand stare everi girl pass pass comment make weird facial express postur sing song brush women pass
gang boy tri rape return home night time howev manag escap
boy screamt train happen coupl time
misbehav
friend rape tuition teacher
chain snatch
chain snatch incid happen near hous stop go hous
report submit safewatch
indrajatra someon grab back
chain snatch near satyam hospit block delhi110089 even
rape sexual assault girl put short cloth pass road boy admir way dress made boy sexual harass
friend follow wistl even boy
ye way colleg indiranagar saw start sing song
incid took place near noida metro station morn saw woman scold small child badli
indec comment pass men street
incid took place around 2 year back valentin near nirman vihar metro station night 56 boy three bike came start circl around friend
go vijay nagar rickshaw wheni notic anoth rickshaw puller stare long make wierd face whistl
boy comment among colleg
bottom buttock grope men group would let turn around identifi
area near nsit campu isol quit unsaf
go mother dairi friend harass guy mani day would pass comment next 12 day ignor came grab hand friend shout kept say objection thing friend learn self defens colleg use work also share incid parent
incid happen front hous group boy teas girl cat call well took pictur girl abl anyth
happen regular basi
unnecessarili touch
saw girl walk way school home boy teas n comment
boy follow enter garden
walk school home saw group boy comment girl walk infront say big hip
ladko ka grop khade rahet hai aur activa se jane wali ladkiyu ka picha kart hai 10 baje raat us rass se humko sef nahi hota hai
guy comment upon girl pretti disgrac even
driver sit pavement made pass ladi intern
sexual molest shiv vihar late night
happen school friend alway fun play men back school level form two villag sex tender age led earli pregnanc
even men tri touch girl comment
man station stare women touch
use group boy street cross daili use teas sing whistl
follow 2 men bike
ladi pleas watch 6 4 man late 20 abus stalk threaten
got touch boy public vehicl
corner street leav school amp colleg boy form group sit comment school girl colleg girl
return school group boy teas show facial express comment dress
chain snatch take pictur touch grope satya niketa
walk around road boy call name whistl
happen twice random biker ask want come first time happen month ago around 730 pm walk forev desert footpath wadala station biker first ask direct ask want come along continu follow till take phone leav second time happen saturday 13th august 2016 1030 pm walk near wadala station biker stop bike right front start use differ facial express thank parent came pick time biker manag get away could complain entir r kidwai road stretch danger scari women walk desert street
come back colleg train boy train start whistl comment
sister stalk guy rk ashram metro station
come back colleg two boy bike gave weird facial express happen 5 pm
even go aunt home 2 boy bike snatch chain push hard
touch grope
pass justaa hotel bunch guy start stalk us comment happen afternoon
go uncl hous two boy comment dressup
walk regular path home group 6 boy start watch everi step start comment look start indec gestur
holi celebr male employe appli colour insid femal employe mouth pour water within offic premis okay report matter
young teenag girl cloth torn street young boy mini skirt wear
take pictur verbal teas
go shop buy stuff way guy start follow tri stop follow came forward took pictur mine
school friend person start eveteas us
2 guy pass train stare pass comment
saw nude man masturb front hous near sicsr symbiosi
molest teacher
friend travel metro stranger grope hand
incid took place around 4 yea back way attend class friend aboy came toward even didnot look face suddenli huge ccouldot even resist helpless moment silent cri friend scold report anyon even parent
man comment return shop home
guy misbehav
goon comment amp whistl friend also threaten us protest
neighbor soon kept ask sexual invit whenev spot street enter hous told want friend happen near som bazaar vika nagar
comment indec action group boy
sexual harass morn man carri polic station rape girl claim drunk girl could even talk moment unconsci littl girl bled dead
group boy take pictur embarras
return home tution around 800pm boy start stalk comment vulgar follow upto hous
afternoon indec exposur inappropri place travel
feel safe crowd peopl gener touch
stalk 2 boy follow hour tri make convers refus talk got hold luckili peopl around
eve teas bu gnr ahm
go home someon bike tri touch back
age 16 yearstyp harassmentstar ogl bu 2015 time even
continu comment bag snatch night hour
pass comment
bu man tri touch public vehicl alway nightmar
peopl stand busstop alway roam dhaula kaun comment everi girl pass
old man kept pass dirti stare sister went market afternoon say anyth told
go back hous car follow peopl click pic car
2 guy rickshaw took rs 2000 made
walk roadsid sudden men bike comment
walk man decid call quotmadam quotmadamquot
night hour guy comment girl pass
 44% |################################                                        |
way school friend stop man start touch breast ran away fear horrifi
boy unknown boy touch
motorbik number bla 463 tri assault juhu gone toward andheri
misbehav
walk road boy comment whistl
wit harass delhi mani time mani incid actual describ action taken male around irrit part
two year ago return home town villag boy follow hous forc give number week keep follow hous
incid took place satya palac street even come tuition guy cycl whistl pass comment
chain snatch
girl colleg got harass morn
man spank aquamagica way saw
live laxmi baug sion shop along stretch station road right upto entranc chawl muslim shopkeep jungle boy throughtout lane shivar guest hous often stare dirtili pass filthi comment happen everyday incid face us girl area even ladi understand author station polic booth area realli danger popul hooligan pleas help outsid sion station tunnel toward shivar guest hous go toward laxmi baug
touch grope chanakyapuri
man pretend drunk start touch girl privat part claim know
go parlour old peopl knowingli tri touch felt nervou
guy stop car front walk toward colleg click pictur could anyth hide face ran away
ill treat
bu stop main road unsaf auto driver tri pull dark peopl around somehow got away ran home
neighborhood junction boy gather make girl uncomfort pass gather
incid happen near colleg
sever girl press crowd metro
friend alway harass guy near toilet complex sanjay camp around help scare
pass vulgar comment make facial express happen even
middl age man open zip crowd metro teas touch girl genit girl along men shout caught hold man beaten bleed profus apologis lot hand metro author
go school shiv vihar notic boy follow ask start grope touch start shout told go away abus slap push fell ground went away ran school happen earli morn
student feel safe even travel morn face assault almost everyday till never taken action ever report polic offici
walk street friend man pass comment verbal answer back
happen train
afternoon kamla nagar
guy beaten badli moreov comment snatch common make unsaf night
watch movi friend fun cinema kota someon sit besid watch porn cell phone
get realli disturb night travel alon road desert henc polic requir street
comput professor touch chest inappropri way
u r cheasi stuff told
friend pass bu park area gang hoodlum comment teas even humili front other
man tri touch inappropri park afternoon
chase follow
guy touch tri press butt
men eveteas walk church
friend school boy seat across road start whistl call name
new guy work chattarpur metro station park precari men stalk want highlight everyon know
stalk man way hostel
stalk man near colleg week
return club night friend guy came stop ask us quotkya rate haiquot rate shoo away follow us
man come grope butt walk crowd time could catch hold
saw 2 men pulsar snatch woman neck piec ear
abus shopkeep said wont give stuff inappropri dress
friend gone beach friend notic group guy click pictur laugh
boy continu follow till reach metro station pass comment also touch hand
come back borivali train guy along two take pictur
touch disgust touch grab elbow hand metro station even
return salon group girl comment say quot see hero comingquot
unknown person teas walk street
follow boy
humili
guy stand entranc duck park enter park made express ignor stop forc start ask prsonal detail
woman 40 statu singl sexual harass driver complain polic refus write fir show evid mobil influenc polit parti fir done polic
guy walk scorpio car tri kidnap girl poor light 7 pm
happen individu leav offic 0800am saw watchman make comment attir independ work ladi ignor watchman left work later union meet report incid union core committe result union made complaint watchman secur agenc time span three day watchman replac anoth
guy random school pass bad comment
man stop car front us start masturb
boy call even follow
travel bu conductor bu whistl take pictur without permiss feel annoy cant anyth els
happen friend scooti group boy came snach gold chain ring
farewel school return group boy start comment us rough word embarass
chain snatch
someon pinch metro
lot comment came way night hour
person snatch chain walk incid took place even
touch grope
finish colleg return home hooligan guy comment whistl stalk home
harass
happen travel dtc bu
man pass sever comment appear bu stare continu
friend mine abus comment guy respond tri touch
stalk
cross road truck came boy insid saw shout quotkya dhunadar maal hai quot cri like hell
sexual invit
whenev go toilet guy stand pass obscen comment make difficult use toilet time
went shop group boy start teas us
comment leg
roam around market shop sister random age man pop ear say quotkya matak rahi haiquot
realli bad
lunch meet coach tri touch breast got mad tri describ childish uncivilis
incid took place near colleg vocat studi sheikh sarai enter colleg walk toward left see wall metal grill colleg friend today afternoon around 1245 suddenli notic man stand side wall saw look nod head way tri call thrice pass fli kiss start shout abus got still abl see face still look us saw two minut disgust stand look us masturb abus leav show desper peopl share experi tell peopl read speak silenc give power pin creep harass
guy bike whistl
friend whistl three men
sexual harass somebodi opposit sex forc want fuck anoth person without permiss happen 1st januari 2015 molyko friend come back church group guy came bush threaten us knive forc us lie could want fortun us good samaritan pass cream help
gurgaon sohna road unsaf night especi alon street light less peopl around vulner
comment catcal happen even
spank guy bike dark
travel hospit home gangal hospit guy follow us thought stop even enter hospit scare
alway men bhaya sit across right urin near signal sion station right staircas take sion station peopl scari pass dirti comment girl pass even coupl get taxi auto auto stand request station offic time 8 pm onward pleas help sion station urin urin sion station stone staircas station
friend daughter rape vacant plot man neighborhood left bleed plot found next morn die result injuri trauma mother still fight case
group friend go bu stop celebr holi bca du pass narrow street broad daylight someon call us back helpless tone though someon need help us turn back disgust scene person taken privat part pant show us even could react ran away happen narrow street near quarter ministri extern affair opposit jm intern school section 2 6 dwarka new delhi
 45% |################################                                        |
travel bu crowd pick pocket tri snatch bag later ttold everi everyon shout ran away
usual peopl sit idl comment girl
friend mine sit auto rickshaw share long time felt someth poke breast area assum man umbrella kept quiet find later hand shatter
guy use follow month feburari use travel coach class
sexual abus concert unknown person grab breast tight afraid
harass
step mother beat make carri 20 litr water beat sufuria small girl mani bruis neck face
40 year old man tri touch travel form kathmandu banepa defend pinch hair pin
saw group men seat togeth call ladi pass school
bu satya niketan anand vihar
person stalk
get back home wednesday market even dark alreadi way back home saw man stare could figur happen snatch chain ran away
walk friend women meet pass nearbi pub man look window start call friend invit go bottl warn man concentr busi avoid admir peopl wive
survey carri safec red dot foundat along safec audit street market mumbai
home town hyderabad waz walk cousin somebodi came ask address stare show pictur nude men women class 7 got scare ran away
ladi cross open field way work man jump bush pant start masturb front alon scare ran away scandal quit shaken incid howev never report
two men bike came extrem close block bike could move laugh zoom
bu guy tri touch area around group boy sing cheap song
sister touch man next bu done intent sister cri bu
travel public transport wit mani indec incid happen even taken initi stop instanc usual take place even hour
man give weird disgust look near public toilet
incid took place afternoon work oye fm culprit boss use find reason touch use make stay work even done work
send vulgul messag facebook account
incid took place villag 16 year old girl first verbal harass boy last rape girl forest collect fodder
bag snatch 23 guy came motorcycl rickshaw happen afternoon
catcal comment bad facial express
stalk
scooti return home work red light near offic ragpick approach made indec gestur first flash also blade mouth soon light turn green sped scooter away scare disgust
guy form group sit near circl keep comment girl walk back township way along friend two guy group kept follow bike kept ask name kept quit ignor continu walk block way back caught hand ask what name push away got hand releas walk realli fast toward township saw secur guard took uturn went guy drive ask could come night rode girl free kind harrass
guy start comment friend start dirti gestur made feel unsaf
inappropri comment
way school certain man removet quotduduquot peni start urin infront shock
man gave bad sign hand teas badli bu
comment
walk sophia colleg kemp corner man whistl drive realli angri hate kind behaviour
propos friend said due distribut number friend start text block number afterward
someon touch afternoon
kurla station dimli lit unsaf
somebodi pinch wait train kurla station platform 7 kurla station
andheri station incid cat call took place
walk back home boy comment ask number take pictur
morn touch
touch inappropri
dirti stare deliber push touch morn even station
two boy stare badli felt uncomfort tabl move place
ogl travel metro
cross road infront colleg
incid occur 6 month ago still feel yesterday shock bad return yoga class 8 normal friend alon walk past church way home 2 boy bike came toward touch stare chain purs hand could snatch time follow stare stood littl look boy pillion seat continu look went away start walk back home reach hanuman templ 2 boy came opposit side bike stop bike pillion rider ask hardwar shop told know could ask someon els shop open hour peopl road said ok start move suddenli pillion rider grab chain fortun grab dress instead use presenc mind lift hand hold onto chain sleev tore process nail scratch upper arm god save chain sped away bike appear 2 boy first bike 2 second bike work togeth tandem mani incid occur road policeman post opposit church
stare
divorc man live besid us custodi two daughter wait 1 year 3 month rape
guy kept follow until reach insid campu
chain snatch
ladi wear short dress heel walk group men stand road men start shout indec dress girl forc run heel fear could undress
harass bu
whistl upon travel even hour
boy took pictur dapoli beach girl
girl come offic 45 boy surround start teas tri physic damag
neighbour attack three men night husband nightshift rape
went shop sari shopkeep touch sister inappropri way help sari
even pass makina stage saw man whistl woman pass
stalk auto rickshaw driver
call talk indec
ye near ground ogl stalk whistl
2 guy bike tri snatch ladi chain coupl day seen comment girl leav colleg
catcal comment
night 9 30 pm
happen month may 2014 wait bu go home back duti wait half hour three four member scatter bu stop suddenli auto pass side gave way stood asid see auto could see passeng suddenli auto stop 23 yard away auto man got start face call mischiev smile got scare see fear face came nearer helpless ladi stand near scare scare pray call god insid silent sudden bu came rush save thank god came home told husband ask note number autorickshaw time mind work fear noth could done time pathet situat ladi
person stalk shop mother pass dirti comment
stare make indec express touch happen gener metro buse even pass bu stand even might victim inappropri comment
friend play guard threw phone chit ran away harass
someon follow ladi street jyatha basic poor street light narrow path
metro yellow line kashmere gate rajiv chowk afternoon friend touch appropri friend feel comfort
young man move pass along street touch buttock tri retali everyon turn said claim virgin beauti virgin mari felt embarrass shock femal joint throw insult
follow 2 biker differ bike
wait bu bhiwandi notic man look lewd manner whenev look look away indic noth happen board bu deliber stood behind allow go ahead knew want touch someway fuck give held collar complain conductor beat drag away
feel safe station face assault almost everyday
feel safe station crowd
bu stop man tri harrass
way home man call peopl pretend call mine differ ash start follow call loud stop ask want find want follow hous
come school meet grp boy schl student carri tiffin sister hand bundl exercis book hand star minut start pass bad comment say carri along stuff
stare bu
saw chhuchepati guy take pictur girl n girl say dont want lition
someon know waz grab touch metro
catcal bad comment
travel public vehicl old man touch privat part felt uncomfort move anoth place
cousin play arena near foodcourt gokart man start film us unawar till uncl went man smash phone
5 year old girl touch privat part young man tri defil
time woman use live mani girl orphan young girl use sex worker use take money old men exchang would give girl men sexual plessur
 47% |##################################                                      |
guy comment iw asgo home
guy ask give money
mother caught rape rasta dam site
goin colleg group boy comment dress style
comment touch grope
go market time two three boy teas girl girl realli seem scari sweat also follow girl walk fastli boy stop follow end go home big issu time
happen kid dont rememb much touch badli happen even
14 year old man tri touch privat bodi part name play hide seek
young girl 17 year sent buy night way back stop gentl man moment ask lover refus decid harass girl
harass train
bu karol bagh boy stare us inappropri manner happen even around 6 month back
experiec chain snatch incid area safe anyon due improp light
crowd bu guy touch act mistak kept
four girl walk saw boy approach us began cal us ignor start insult
even travel even wear simpl tee jean guy lecher eye turn back till time comment start touch
saw man tri touch woman breast insid pack public bu
noth much happen still secur concern
happen anoth passeng harass dress wear two guy pass comment
travel back home sinc lot crowd road side get rickshaw henc decid walk call husband pick till come instead wait decid walk start walk thane station cross alok hotel backsid road group young drummer start follow first ignor later start pass comment could anyth decid keep walk silent follow till husband came see husband ran away stop pretext smokethan station road near aalok hotel
cellphon got snatch
friend went girgaon chowpatti walk group boy follow initi thought coincid soon realis follow us decid chang rout
chain snatch
sexual harass school teacher teacher use touch privat area abus sexual
condit realli bad actual difficult walk around enven nigt
survey carri safec red dot foundat along safeti audit street market mumbai
overheard comment pass girl metro station happen even around 6
night chain snatch bu
eve treas wistl say unnecessari stuff time touch
boy nearbi slum whistl
rel touch everyday feel harass
peopl comment filthi comment afternoon
travel bu saw girl touch boy privat part
walk toward olymp stage take bu men work weld shop strted talk low tone turn look
afternoon walk friend mine met boy corner start comment friend
friend eveteas passerbi bike took round pass us sever time
usual colleg way back hostel local guy call chinki
sexual invit
person follow sister pass comment tri touch dongri sabu siddiqu hospit shalimar
realli bad
month ago man make advanc toward touch breat went buy drink discuss becam angri gave slap face becam angri want beat peopl came ask problem said insult imagin
stare unnecessarili comment weird stuff
felt harass humili take pictur pass comment
rickshaw wala tri take girl wrong way call friend come happen morn
touch inappropri place
happen last 30 day night way home notic group 45 boy sit corner girl cross way guy comment ignor went
disturb guy comment stare
guy pass comment tri touch back happen around 7pm
group guy follow friend way school chang rout
13th cross 8th main auto rickshaw man ask sexual favour show money could understand first languag problem later understood show middl finger
sister go shop buy cloth walk encount group boy start teas sister comment invit movi embarrass could take stand avoid walk along
misbehav
misbehav
even tri touch bu stand
stare
car uncl sister alongwith uncl friend go villag uncl touch inappropri place
biker pass comment ask come happen even
misbehav
boy group start stare hoot ask person inform comment bodi figur
16 year old marri young boy 25 year agre go school hous suprisingli want us start children wish brought seriou problem tear pant want us sex even unsaf period
travel public vehicl guy touch privat part
n eye mani offic pakistan fill lecher boss often work women tale pervers lesser one includ gaze glanc occasion grope unwant text messag innuendo bigger one includ invit meet outsid offic lunch dinner plum assign promot job secur profession reput hang balanc resign guarante repriev refer letter obtain futur employ worri expens inflationwrack increasingli competit pakistani workplac mani women continu target men power arithmet want need display divorc women singl mother older unmarri particularli vulner harass word singl mother endur 10 year harass pursuit constant attempt escap punish denial promot humili colleagu cowork often wit say noth eager avoid situat could result retali loss posit sexual harass superior henc often coupl isol colleagu watch wit withdraw harass persecut also pariah law pakistan protect harass women workplac act 2010 six year old thorough document code conduct includ defin harass unwelcom sexual advanc request sexual favour verbal written commun physic conduct sexual natur sexual demean attitud caus interfer work perform creat intimid hostil offens work environ attempt punish complain refus compli request made condit employ goe add unaccept behaviour organis workplac includ interact situat link offici work offici activ outsid offic abus author creat hostil work environ retali three categori mandat action harass detail stipul set establish investig committe ombudsperson etc charg resolv issu rais rang penalti censur outright dismiss postul employ requir display code conduct promin premis law pass 2010 fete success would take time provis chang cultur workplac circumspect said chang come slowli law first step legisl commit support elect repres signal path ahead differ case harass women workplac would permit much happen sinc harass still rampant workplac major workplac littl idea code conduct let alon requir display visibl employe area women still regularli report verbal harass even physic assault superior make calcul regard need job desir get ahead inabl refus unabashedli continu act face unwant advanc pakistan women continu find alon unsur take complaint protect even develop sector agenda mani organis empow women similar problem persist cofound organis face much harass male colleagu ultim forc new pliant women hire take posit sinc complain similar problem man question howev remain untouch undoubtedli display similarli harass behaviour new prey men come defenc perhap recognis behaviour other eager ensur get punish misogyni manufactur two major flavour pakistan first premis religi obscurantist whose hanker reinstat strictli segreg societi see harassmentfil workplac grim substanti warn women workplac male conscienc unabl polic second wrongli label liber progress imagin mean licenc harass harangu woman willing put shut mind product progressiv illog mindset equat women public sphere women sexual avail men may want two flavour compet poison infect work live women doctor lawyer shopkeep banker teacher professor countless other daili forcef bitter morsel misogyni law alon chang societi sexual harass women nearli everyon read articl either know someon someon face harass continu consid permiss someth women ask leav home belief reflect time everywher pakistan soap opera vilifi work women predat seduc innoc men workplac convers men dissect desir femal colleagu particip often measur masculin fear competit women matter sexual harass pakistan workplac guilti guilti far mani men complicit quiet eager embrac enforc right harass
comment mehram nagar near delhi cantt even
harass
go back home around 8 even stranger forcibl held tri molest use certain cheap languag tri convinc intercours session lucki enough break free flee
bu man tri touch felt uncomfort felt like get neg vibe
guy work colleg canteen take pictur make video friend morn hour
age 15 yearscom leh bazaar shop time
misbehaviour guy
guy follow till home misbehav right front hous
night guy tri touch
say go launch us insid hotel
touch
didnt wit commun 2 men bike snatch chain ladi dash happen night
fetch water girl wear short dress came fetch water men start question dress girl scream strip
stalk
girl come school met man drive man call promis buy anyth need even gave contact
take son school morn met old man wo touch buttock ask want beat till ran place report
even time return tution face comment n
realli bad
group young boy cluster togeth group usual cat call comment women pass
happen main market group boy pass start comment
teacher touch privat part left toilet
car invit hoot sit car
incid took place colleg pass corridor group boy use pass comment teas situat discourag walk corridor
two men bike tri snatch dupatta pass lewd comment resist
happen around 730800 pm wait train come platform laxminagar time boy tri touch comment
friend mine abus man market pass dirti comment tri touch inappropri
sent neighbour grandpar ask littl salt neighbour saw want touch ran
walk 8pm ahmedabad behind gujurat univers man follow space street grab hand shout loud possibl run away polic station follow
 48% |###################################                                     |
group boy follow us ask us number
comment pass group nearbi ticket counter
outsid metro station auto rickshaw valla stand liter hold peopl hand let sit vehicl
rain 800 p stranger gave lift sat behind two men pass alight point ask loudli mouth cover two men rape drop kangemi
heard lot man say bad vulgar word public place even tri touch bodi part whistl
incid took place even year ago person expos privat part invit sexual activ person car stop direct sit nake car
walk men say boy stare comment happen even
teas badli boy street
boy touch felt bad go outsid home
outsid shop wait someon man came behind touch comment someth walk away
harras even
conductor tri tri touch bodi girl insid micro bu
travel crowd bu guy stand close tri make attempt touch inappropri defend move forward
man follow us andheri station bandra station
harass afternoon
lane chawl narrow boy stand outsid lane give way women pass pass comment sing song well also group boy pay carom road
follow man
enter metro someon touch butt breast
happen afternoon hour travel metro peopl age group constantli stare comment
survey carri safec red dot foundat along safeti audit street market mumbai
boy shop stare comment girl
saw girl wear walk road man stretch skirt went away
man grab dupatta tri touch breast dark happen near balaji chowk
return home work public vehicl man sit next kept hand thigh amp tri get close
boy pass bad comment friend
chain snatch
man sit along railway line told us old continu school
touch grope
luci physic diasbl child assault stepmoth get physic injuri child rescu taken safe hous
peopl touch grope khot lane comment stare
way hous kanhaiya nagar metro station group govern school boy came near made group around snatch headphon hard broke start comment happen around 3 pm
went take tution teacher start talk weirdli alon sinc miss class teacher tri touch forc somehow push ran away sinc hardli go quit difficult forget almost year sinc happen say anyon today write hope bring chang
group men pass lewd comment cloth
blind person ask girl show way toward platfrom 4 bt girl told way tri touch wrong manner
friend call weekend go reach guy get weir that told want sleep gay
inappropri amp danger behavioureven 7 pm
touch grope
place himayat nagar hyderabad men usual three ofcours law breaker ride bike pass comment like quotsexi ladyquot quotdarl ride kaavala need ride quot pass close pedestrian girl stare right face smile scene common traffic signal 2 occas men gave fli kiss happen broad light well night fall ocass 2 men stalk friend 1 hour tri lose would know resid eventu run aforement encount happen within last month nov decemb 2012
man go work earli morn met group wome seat roadsid sell chapati bean start talk put andmock sexual man blush
man tri touch privat part behav accident hit
man pass dirti comment bdi
16 yr old boy announc class go rape teacher femal teacher complain touch bottom breast even say follow laboratori
sister follow unknown person bike
happen afternoon even
auto 2 boy bike start teas laugh safeti remov cell phone start act note bike number move
stalk sexual invit
event happen raini walk road around 7 pm even friend suddenli man cam behind touch wrong place ran immedi could react gone far away fear walk street incid left word take step stop
come back school friend boy scooti slow scooti speed felt awkward decid ignor walk distanc saw boy park scooti along road side constantli stare start pass comment time instead ignor warn complain incid polic stop
boy make cheap comment
friend walk back hous guy walk lane fli unzip masturb openli enter hous approach ran hous quickli could say anyth
mani incid grope public transport
catcal
walk platform guy cross squeez butt got angri gave slap
bu boy age 28 30 approx tri touch shoulder felt awkward drop bu
go travel bu person touch shoulder
guy comment tri talk ask number
sexual invit guy maruti van pass
comment
survey carri safec red dot foundat along safec audit street market mumbai
boy insid car comment went away happen afternoon
afternoon
take micro bu arround 530 630 micro conductor touch breast ever screem
chain snatch wallet snatch eve teas
public vehicl boy teas scold
man morn tri verbal assault
wit chain snatch event area ladi stand wait auto two guy bike came snatch chain left bruis injur
even comment upon
night girl saw somebodi invit sexual intercours ran away
boy stare us tri use phone us doubt went took phone click pictur us
walk home man walk toward actual slap bodi look angrili simpli said sorri ran away near arab gali two tank grant road
chain snatch ramja road karol bagh new delhi
bad boy friend friend told
catcal ogl
peopl local theater manner went fri watch movi guy sit next tri touch hand
went see friend palc happen meet brother start tell stori wait next thing saw brother lock door rape
group guy bike pass lewd comment hoot go friend
afternoon 2 pm
harass nangloi jat peopl ogl chain snatch
realli bad
stalk local shopkeep month
public tap morn sister want fetch water hous boy sexual harass put hand buttock
bu man sit front tri click pictur mobil phone ladi sit next shout see
stand balconi 900 pm whan guy come closer societi dark nobodi start peel pant horribl experi realis masterb
biker came comment abus
group guy stalk wagonr tint glass
shop child 1415 year age sit mother suddenli garment fell land child lap shopkeep tri touch privat part order pick shirt look intent
thought hyderabad progress citi crowd area hitech citi safe commut often use public transport commut felt quit safe men ogl live center area necessari light traffic road start go jog lot men found morn late even real face area came right passerbi grope men pass lewd comment common gave week
blind person touch bu made feel unsecur
go colleg morn time man age 3035 misbehav use vulgar word
get dark night seem danger
two friend two guy late teen wereclick pic anf follow us
govt school boy group touch back park feel unsaf go hous alon
school around 700 pm walk across terminu group seven boy appear took us thin coridor start touch us place street light
man made discomfort express touch indec local train
guy mental retard girl area manipul man marri parent complain later found gausala rescu
ladi walk slowli high heel shoe muddi area look keenli step men start enjoy say quot walk type shoe wear ladi utter word look funni eye walk away
friend go colleg touch harres group boy
head home school saw group men sit insid kiosk girl whistl men pass whistl felt uncomfort
ye galli near templ someon pass comment stare
today climb stair railway bridg man pass say babi make cring heard wors thing
 49% |###################################                                     |
stand queue ride boy stand behind take advantag push forc tri grope
children ridicul older boy make fun regard dress sens financi condit
sexual invit friend
father wait mum left earli market defil
call chinki charact less 2 men road
often even nearbi men guy hangout shadi area often cooment stare
morn 1000am travel bu man behind continu tri bring face near face hand
continu comment whistl sit alon wait friend morn hour
auto three guy came bike snatch friend bag process fell move auto becam unconsci hit head could rememb anyth
walk ride scooter peopl comment us
walk colleg guy startedfollowinhg
walk rider stare whistl come close etc
go lone street two boy comment quotooo bhai bhut jyada garmi ho gyi hai quot vulgar tone
go aunt hous alon bu came man start talk ignor first talk rude complain aunt bu scold
two boy tri lure gave cosmet item take pictur public vehicl
go back place stalk guy shameless came place mum open door said want marri daughter even know beign stalk regularli polic interven took away
robberi near hous also tri rape molest maid present home
go colleg boy call wistl
touch ogl
friday even come back school around 730 pm group guy start follow call sort name whistl
two boy scooti pass friend say quothelloquot treacher manner
white tata sumo fill around eightten men halt suddenli three women cross road shout us aggress demand know thought repeatedli go extent cross highway divid follow us till bu stand driver kept engin run group insid make loud comment also made threaten gestur open door continu run vehicl besid us final backtrack panic ran 100 metr back offic
comment pass wait bu
woman undress boy claim indec dress
two men follow way school street live wait go away get know live
guy made indesc express return home tuition
return back school micro bu pack man near seat tri touch
girl nearli rape went shop alon night adboc girl walk alon night talk strang peopl
coupl ago incid backsid home open ground somebodi came stare window bathroom scare suddenli saw smone look window hide day repeat act realli got frighten
drunkard friend assault sexual got involv gang fight
misbehav
girl made feel awkward tri enjoy park
morn wallet phone stolen
sister goe colleg group peopl stare pass bad comment even tri touch cross whenev possibl
whenev go toilet complex harass
travel rick besid guy around 1516 year sit next cours travel tri come close freak verbal respond
alway touch travel public transport
follow market first thought imagin sure
hi women safeti want inform nagpur citi maharashtra india talk particular area talk rout like railway station main road etc anyway talk koshtipura sitabuldi nagpur everi street boy servant shop make alway joke girl treat like prostitut treat everi girl use abus word request pleas take action area thank
saw man squeez young girl corner even
ogl comment
near hous small grill factori employe male go shop boy teas even write rough word wall van
follow school street live 2 strang men thought rape scare
travel bu teenag boy whistl comment nonsens
around 6 month back follow guy even
chain snatch road
harass call repeatedli say kill accept propos
walk road two biker snatch ladi chain
happen last 30 day night way home notic group 45 boy sit corner girl cross way guy comment ignor went
girl come night shift two boy bike push shout loudli
harras eunuch
comment outsid metro station even
stare touch
subject indec remark sector 21 nerul close home
catcal near bu station way shop friend
misbehav
incid happen near zoo girl return tution class comment badli boy
misbehav
incid took place rajiv chowk metro station insid metro boy pinch badli waist hit back sharp hair stick
tri touch breast elbow hand
happen afternoon even
shop wait shopkeep man came nowher touch butt embarass
felt someon touch attend tuition class
touch breast man bu said mistak liar
come back colleg privat cab soon reach stop driver said want call friend forgot mobil place gave mobil help call save number next onward use get call messag 2 day told dad within week switch number
area mark near slum area guy dont drink home set camp bike park street drink right cop come socialis leav guy talk loudli laugh stare women pass
sexual harass first sexual harass teacher former school call offic ask name address told first friendli last long break call offic said make succes alway take first alway good grade condit sleep refus said deal other done regret other slept enjoy day later gave first test zero pain never knew went expos problem friend greatest surpris said first girl school sleep teacher good grade confus decid chang school second harass uncleh relat us holiday came visit us sinc hous tight forc sleep thesam room forc take alcohol thank god drink alcohol went threw night everyon asleep start touch raom hand shout everyon came
famili monasteri call monk room talk interact class buddhist principl respect well known monk room tri touch bodi uncomfort way pretend care later told parent said remain silent would believ
incid happen aunt travel bu first harass touch shoulder said anyth time touch mean time bu conductor saw harass got scold conductor respons aunt also request driver take bu polic station file complain guy later guy shame
boy whistl follow
afternoon comment whistl upon budg even warn
touch travel bu even hour
man bu sat behind inappropri tri touch howev friend rais voic thrown bu
micro bu realiz girl men teas conductor came close seat could anyth yell got bu
catcal comment bad facial express
someon know wink creepi man made nasti facial express disturb
walk man came behind grope second went give long dirti stare
man whistl comment
travel back dlf porur bu crowd guy intentin press back yell move place follow got 2 stop ahead actual destin realli wish end asap
sister go cinema hall two boy came sat near sister start teas touch
sexual harass happen friend pain incid come back night club group boy attack entranc hous took bag money phone boy start beg undress could resist final rape
whistl
man around age 0f 50 year touch inappropri wave pool fun food villag gurgaon
come back home school friend afternoon boy pass obscen comment month
catcal bu male passeng
return coach stranger com bike touch
boy regularli use stalk went ask problem said noth left next solut
guy tri touch girl bodi crowd public bu
go friend snack stall outsid colleg guy start pass lewd comment us
return back colleg found group boy teas sing everi girl pass side
shop apparrel shop name adam year ago use trial room care tri cloth cloth alreadi wore still afraid dress trial room caught camera year later see guy full chillout band kerala make offens song quit intellectu indirectli make feel much embarrass proof except comment post facebook clearli furtiv point toward
return airport 830pm boudha follow boy bike follow upto school ran fast could
receiv second verif code email phone peopl like use mail irrat peopl
stare bu driver
stuck crowd thursday market low cast cheap touch inappropri 3 time breast ass abl find first crowd found shout nobodi came help pretend heard noth sadli ran away could slap
boy follow touch
somebodi pull hand walk street
misbehav
stare comment near colleg make student feel unsaf
 51% |#####################################                                   |
crowd place 1 person said someth disturb quotchut chatn degiquot sinc crowd place peopl push could react instantli time realis happen alreadi left otherwis also know would react even still stand
walk toward olymp stage take bu men work weld shop strted talk low tone turn look
go school morn give exam mani peopl around man came nowher start pull skirt start shout kick slap ran away felt humili skip exam went sat park alon
comment gone buy milk nearbi shop
night 10pm pervert continu stare check girl
ground near indira nagar comment whistl ogl
2 girl danc suddenli 5 guy came amp start touch amp abus rpm vasant vihar
touch felt man
group guy whistl ogl comment take photograph without muy concent
friend stepmoth care much brother use assault mother knew still say anyth friend stood shout front societi serv 6 month jail time
man kept stare bu station took auto left
gang boy come hous gairidhara area rob hous show knife loot jewelleri money etc
ill treat
boy tri touch girl couldnot thing
group boy threw flower comment indec walk ahead
walk even guy car group start shout whistl
two peopl came bike back side woman wear two gold chain snatch boy sit behind bike ran away quick possibl
colleg presid 50 year old man took power author friend
morn walk guy eve teas start call name comment happen 23rd august
work place ignor time later brought knowledg concern author
malviya nagar market boy teas friend
pictur also taken complain polic
two wheeler two boy snatch chain fell scooti
group men stand around tea stall main road block keep stare comment cloth women take pictur also sometim
front build come home rickshaw driver honk make dirti facial express
father sent shop shopkeep told enter help carri packet milk sit room tri rape son knock door save
two guy scooti pass front cross lane whistl said quotmeowwquot
touch sexual invit scream push
holiday play friend certain boy came told call certain man went reach room found man nack close door forc thing start cri make alot nois got scare open door ran away
man seen lot girl face problem good
chain snatch touch grope
colleg boy next faculti use follow friend unnecessarili amp give nonesens comment make mental disturb share incid complain matter coordin mam seen person
man grope public
ogl touch grope
rajiv chowk even time face comment road side
friend walk school home peopl comment us bad word react start fight us
wit guy stare girl even follow

famili friend tri molest daughter
incid took place dhaula kuan subway even around 7pm cross subway held back group drunkard tri sexual assault escap help friend rebel physic
follow whistl chang daili rout
middl age man tri sexual assault knife point
happen around 8 clock
walk way guy start say quotkiss kissquot come talk made uncomfort
went visit male friend discuss upon return grab forc
saw two boy bike come opposit direct snatch golden chain middl age woman cri loud get back vanish second place
even underw catcal comment stare
almost everyday face
4 year old girl found miss amp murder
stop peopl claim work satr time compani sell digit decord said sale promot readi give scratch card took scratch behalf said 3 cell phone lap top get give 5000 told money insist call someon send phone account refus peac begun comment bad dress touch buttock say could chang wardrob
bake bihari templ extrem crowd touch inappropri
harass
sexual harass home work househelp order earn live harass took place night druge rape master
go school public transport space stand sometim found man hold behind know pinch badli gave bad look
catcal comment touch other area
walik street polic stare girl gossip nonsens abt tht girl
even travel group peopl constantli stare comment stalk
touch bodi part
comment ogl
dadar station main bridg crowd guy touch inappropri yell cop bridg tri help peopl around cooper could find
guy follow friend tri hold hand
follow man could see properli
happen afternoon bu bound toward uttam nagar termin
afternoon hour touch face made
group men pass lewd comment women pass also engag whistl laugh sometim even touch femal commut
two boy tri click pictur bu stop ask start abus
felt nauseou due excess cigarett smell
outsid metro station tri press boob bu abl tri call back nobodi like total isol
crank caller continu call text via messag inform father file fir mobil number onward get call text guy
two girl stand bu stand gang boy came bike start teas whistl toler minut bu arriv went otherwis incid would becom seriou
embarrss situat
friend stand queue pay bill ogl group boy
happen overnight goapun bu travel alon somewher around 2 notic guy sit next pretend asleep constantli fall shoulder touch inappropri
survey carri safec red dot foundat along safec audit street market mumbai
comment girl pass rohini stare board metro incid took place even
touch grope
go toward class 2 guy yell someth abus comment someth
often experienc men take advantag crowd place brush women bodi touch women inappropri way experi friend
survey carri safec red dot foundat along safec audit street market mumbai
boy tri touch friend street
poor light mankhurd railway station
happen friend dda park afternoon get back home school exam 34 boy around start whistl comment told come back night take away
group 8th grader start make indec sound group north eastern girl pass happen even
sexual harass night
afternoon car start follow travel riskshaw order get home stop till reach home
age 16 yearstyp harassmenttak pictur 28th march 2016 agl leh even catcallswhistl skalzangl leh market 2016 time comment school leh time sexual invit leh market time other ask phone number even
incid took place afternoon public park dilshad garden sit bench next start unzip pant ran place
follow
half age man tri touch everi small girl micro bu seat
incid took place around 330pm group boy comment stare like bastard girl
househelp neighbour almost rape come market escap
chain snatch teas
comment catcal whistl touch
group boy take pictur mine tri touch bu stand
teacher known teacher mark defil pupil took book mark child report talk box
parti job friday late come home alight makina stage took motorbik take home instead take home motorbik driver took friend intend rape screem help process struggl tore cloth ran away
small girl 5 beaten man thick stick near kamla nehru
shop keeper pretext show bag interest kept touch breast first thought accid within second realis intent immedi turn leav kept ask quotkyu madam acha nahi lagaquot give disgust smile alon shop friend anoth shop angri could even speak sentenc made sens shout left smile like done noth wrong daili
metro guy tri touch inappropri manner
incid took place 15th septemb 2013 around 830pm friend go peopl start comment
face verbal non verbal sexual harass colleg way home other
want get physic metro
street festiv boudha street much crowd got harass
 52% |#####################################                                   |
group person sit car follow rickshaw tri talk girl sit rickshaw pass loud comment
random guy touch breast walk roadsid
gone nss camp 7 day use travel school morn everyday came across bengali buzi mobil walk girl notic take pictur complain author went speak later public gather phone check confess took pictur leg chest part etc start hit men later even file complaint polic station phone taken custodi
person train tri hold lot crowd could help chang place stand
girl sexual harass dirti south night happen room victim serious harm taken hospit good heart peopl came time
lack convey improp street light
misbehav
follow metro
girl teas boy comment leav work
boy tri touch girl public bu
wait metro guy stand behind comment stare
girl follow bu stand upto home comment know total unknown pass bad comment alon could fight back
travel micro person tri touch girl
beaten snatch everyth includ book mother deni food sent away mother
physic harass amp sexual invit
need go mri scan rel upsana hospit need chang cloth person open door mri scan ill person door close forget lock open purpos old men stand next fill crowd
go shop friend boudhha mall group guy unknown boy boy look star make feel insecur scare later start show facial express blink eye n feel helpless
walk street boy 45 besid start sign whistl later laugh comment
ill treat
return coach boy stalk
realli bad
eve teas bu morn
student way school decid totak short cut reach school earli way certain street boy attack drag bush serious rape three boy slept helpless passersbi saw book trace found lie helplessli
go home school saw girl teas boy girl feel uncomfort
hoot comment
group us three return visit famou monasteri white gumba public vehicl two boy follow us place gongabu
morn comment upon
attend parti happen get around 1 could manag find rickshaw travel back home alreadi consciou travel alon fear rickshaw happen look drunk drive fast despit warn ask stop kept drive ahead shout told aggress stop place got ran quickli without even pay incid made huge mark
frequent travel mumbai local train like mumbai women take harbour line train goe mahim cst seen mani time 8 pm train get empti presenc rpf personnel railway look increas secur rout
lot comment quotchamak challoquot afternoon hour
man pull skirt ran away scream
two boy colleg student presum follow bunch girl whistl time left motorcycl
scari experi 2 guy stalk pass comment took auto night
boy continu follow till reach metro station pass comment also hit hand
take pictur whistl comment stalk
guy
ladi invit room cours neighbour got nake invit sex
stop man road tri touch breast
indec comment almost everyday
go templ man came push
teacher beat pupil
work friend go home way boy night friend alon boy start teas whistl felt uncomfort ran away
boy grab hand
boy tri touch girl comment bad word
travel bu hear girl scream everyon look back slap boy said touch thigh without consent everyon stare matter got seriou went near polic chakrapath
56 sexual harres offic time
guy bike would let us pass
incid occur three month ago guy brought girl tiko sexual assault guy knew bodi area knew neither girl stori told passersbi heard girl scream vehicl go saw guy forc girl open door guy excap ask guy girl respond even know guy
park murder happn due theft chain snatch park safe peopl
person famou wellknown social worker commun write messag everyday fb inbox say love give differ kind gestur
night tri touch also kiss
harass
harass
friend boyfriend tri touch privat part
wear red dress go friend hous boy sang song tell red dress look good
men stare poor street light
ogl facial express
get back shukar bazaar even man start follow whistl walk along ignor quickli got back home safe confin feel scare go alon
park nearbi colleg peopl use harass sexual happen time
saw guy take pictur touch metro even
boy colleg took pic n made viral around colleg boy group came know friend n actual boy like afraid lot yr
group boy start comment told pictur shout peopl gather took away phone
two separ instanc 1st happen last even 5 30 ryt next templ indic lone area 2 youth bike tri grab ladi chain fail went ahead turn n came back n snatch bag n got away 2nd instanc street around corner meter ahead laptop snatch young grl
happen near sreeram ia institut
teacher beat student
group guy coach class use ask phone number
boy tri snatch girl travel
friend pass mashimoni area men normal sit kiosk near railway chew mirror cale whistl
realli bad
even travel whistl stare head toe
harass afternoon
take pictur
colleg samarth colleg stare
head offic afternoon time man walk opposit direct walk toward pass dirti sexual colour remark react immedi yell argu back came hit push back left commut around none interven offer help tri look polic home guard none around
anoth girl usual disturb boy around rea left school sent away mum suspect girl sleep around boy favor
school boy alway steer amp whenev go somewher alway follow wheather tution amp streetani time
boy took pictur comment unnecessarili
touch unappropri
three year back friend travel best bu dadar station man came sat front friend frist stare longest time expos privat part discomfort friend got went elsewher stand
man sell sari red light refus buy call chashmish vanish sight even
random comment girl
tri forc kiss
walk home friend group boy start call us ignor start chase us us contin abus us
stalk make feel uncomfort
murder
boy si doorstep pf shade ent shop sell cd comment girl pass walk along road
chain snatch
walk school man offer pay bu fare make facial express toward
boy road side say chalti h ky noth feel secur took auto n move
om sai beach hut safe femal solo travel owner gaurav bhagat drunk told staff watch see sleep friend stay seper room call check entir staff give inform activ whereabout bother hit eventu staff watch report make feel safe fact mani doubt overal safeti place hut bathroom hole wall someon could see insid overlook doubt someon watch slept chang cloth thing requir privaci recommend place anyon know told
realli bad
know someon sexual harass report manag final left organ visibl action taken senior manag
start minut step hous loafer tri walk close u mutter someth breath say someth beauti look aloud sever attempt talk back help peopl around interfer becom sheer mean entertain
mani youth drink mountain area throw bottl make difficult women children collect wood forest glass bottl lead injuri leg
peopl take pictur stand area wait friend
five boy snatch chain girl ran away
girl surpris let us comment
touch old man sit next micro bu come bu colleg old man touch lower part everi person seen nobodi spoke word littl
physic abus travel public vehicl
happen near select citi walk school student irrit girl ask phone number
walk across street cp boy came bike snatch dupatta hurtful come outsid hous 2 month depress
happen mostli buse come colleg dtc bu guy indec expos privat part realli disgust happen afternoon
friend return school pass comment group local boy
usual harass stare someon teas someon
comment
age 16 yearstyp harassmentfaci express leh market 2015 time take pictur leh market 2015 even catcal school leh 2016 time
friend eve teas
ogl facial express comment
 53% |######################################                                  |
gone market eat someth sister even way back guy start follow us ran home told mom blame us invit upon
lot crowd guy felt friend
go work saw man neighborhood approach ask name refus block way kept stare push went away
stand friend eat ice cream group boy came pass us went littl far stop start stare us minut left place
bu man purpos fell touch inappropri excus
comment whistel afternoon
survey carri safec red dot foundat along safec audit street market mumbai
went meet boyfriend colleg group men stand near gate colleg probabl find attract follow till reach class wherev colleg happen whole follow even pass comment bodi shape
wit 9 year old girl get wink 22 year old
guy pass hideou comment girl dress pass nearbi took place even metro pitampura
friend stand somewher model town saw somebodi click pictur although sure click pictur express made quiet clear
due natur victim job make come home late ocass gang boy corner beat rape collect phone purs recognis search boy luck
chain snatch guy make absurd comment indec facial express
walk toward metro station colleg boy bike pass comment
catcal comment bad facial express
touch grope dtc buse afternoon stand bu peopl rub bodi stand offens way
cross signal man sit bike call thought ask help went ask fulfil pathet desir
grope peak hour rush dadar station bridg
boy take video back walk street
near paschim vihar east metro station usual dark road ill lit men seem take undu advantag either chase touch behind happen even around 6pm
wistlingsing bad songsstar
wear parti dress guy street whistl
harass even
mail friend guy stare want invit intercours sex
day back saw incid chain snatch front ganga ram hospit
kbar main shyam ke wakt market mai ja rahi thi tab vaha bahut bheed thi tabhi mere pich ek admi chal raha tha usn mujh gandi tarik se chua
came mall around 10 pm walk friend suddenli two guy back tri snatch friend bag luckili hold real tight
would like share year ago manipur peopl behav like normal
throw small stone girl comment whistl dhaulakuan
bu go colleg sit aisl seat man came start ruf crotch shoulder
go coach man stalk
taxi driver suddenli stop dark road told wait went reliev felt scare vulner
man touch inappropri move away immedi
boy drunk follow near bu stop kirori mal colleg return pg attend function ramja colleg
harass commun indira nagar
gone photo studio get passport size pictur click guy told collect next went next told beauti pose pictur refus start forc push ran decid tell anyon would blame
go shop guy follow
touch back bu pack mani peopl know
guy whistl pass cheap comment
harass bu driver
boy touch back inappropri way happen night friday market dakshinpuri
happen even
touch boy
walk across bridg friend shove push stranger grab hold chest
wheni microbu man show privat part noon axcept two us bu condustor front seat
guy tri molest tri take car
stalk
go back coach hme group 34 boy stand comment wildli ask accompani
harass abus special counter ladi renew first class pass man behind want buy ticket wait behind refus abus tri push also tri stalk borivali station took pictur start make obscen gestur
group hooligan click pictur friend approach ask stop abus us
weird area mostli low incom group
group three guy pass comment girl shop almost corner happen afternoon around 23pm
man 3040 tri touch privat part hide hand behind bag
guy pass comment like quotchinkiquot laugh happen even
somebodi pichn backwhil travel train
stalk stare grope sexual invit
harass road
catcal hour
guy unknown keep follow ask kiss
girl go tuition mornig time middl age man tri touch n slap n ran awi n never use way frm next
matatu langata men board matatu sat besid kept star felt uncomfort could tell thiev allight men told noth would want evrybodi els laugh
friend mine rape abandon bush school children met lie earli morn helpless taken hospit well wisher parent heard hospit
walkingon street boy comment
friend went carter road fun suddenli friend notic guy take pictur us went told show us phone found pic delet slap man walk
friend stop 2 boy forc kiss
dark alley man approach back touch
colleg batch mate use sing song comment figur invit date teas us
random stranger came made horribl comment cloth
happen north campu afternoon go rickshaw group boy follow categori select
encircl group
incid took place septemb 2009 even fate lot crowd much hustl touch grope
small girl haress old man local bu
student studi senior colleg friend face tortur hand boy colleg boy affili polit parti highli influenti call name pass lewd remark name sexual messag scribbl colleg wall suspect resolv problem
comment whistl
realli bad
time 12 night sandhurst road girl gang rape know speak boy support local corpor
way colleg comment took place push dtc buse
colleagu report follow 4 men car start maruti suzuki build ambienc mall nelson mandela road
class ten friend got sexual harass although studi school share incid went toilet boy follow toilet unawar enter toilet tri molest toilet manag run toilet later told other manag punish boy taken action school administr
regular comment stare
guy tri follow us silent start pass comment later along way happen night hour
incid took place 22082010 around 9am group boy comment travel bu
happen morn afternoon
guy pass comment group girl wear kurta
touch micro bu
man call hijra said bad thing
23 larg men came walk us ask name teas follow
feel annoy hear cheap comment stanger
drunk man comment tri take pictur station
observ kind harass insid movi hall girl deni guy forc touch
peopl stand along road stare peopl
guy kept stare reason
misbehav
wait bu bu stand gang boy came realli close start comment sing love song
misbehav
go parti friend nice dress want get parti sooner took short cut dda park 3 men sit near entranc follow us misbehav us touch friend inappropri somehow got away
afternoon travel continu stare ogl
sexual harass grandfath call help homework unawar intent 5th grade incid help cautiou other press breast
 55% |########################################                                |
girl shiksyadeep colleg go hous sathghumti bato road long peopl around come near hous 5 street boy rape helpless time
girl teas go home
call new age chhokra citi tendenc accumul near lake side street kick two wheeler vehicl ridden peopl especi women
harass lane commun cloth also pull
morn bu boy stare continu 2 hr
face person whistl give oddlook walk lane
eve teas near mc donald sector 14 dwarka
girl came back morn walk harres old man bridg
man stand middl road masturb
pinch return colleg bu 29c two stop told exactli word english could slap hand full bag back also 29c also pretti crowd even enough space anyth wish stop
wait bu found 2 guy stand littl far first act click selfi point got know click pictur quickli move away
boy reurn home school girl comment look
guy whistkl
touch someon inappropri zanskar sani nasjal
happen 20th may went even enjoy sudden two guy came forc direct scare follow reach hous forc sex guy
even agroup boy comment girl make non sens thing
go colleg morn guy stare top bottom comment quothey sexyquot
follow old man
common mutengen even around 5pm decid take walk met guy stop three surround 3 knive order give phone money watch refus threaten stab went along money phone watch
boy come near pg forc girl say quoti love youquot
way matern home ask give number need compani etc
comment
cross bombay street peddar rd comment pass cab drier biker
guy hold chart intent touch
walk club friend two men van catcal us stop traffic light
incid happen twice area man period two year januari 2014 mention man slightli chubbi bald guy cross path block away home look decent use bike realis licens plate half broken absent cross came back stood opposit side road peni stick pant ask take look masturb shock react time taken back recent man balder fatter cross path road time near build man bike nicer look black red sport bike worn short shirt peopl wear gym black came back maintain safe distanc time said derogatori vulgar thing privat part along want see trigger memori 2014 frozen second react escap bike seem like serial molest modu operandi target girlswomen area wee hour morn bike plate certain live area
2 guy bike hit girl back snatch gold charm
around 30 day back go tuition madangir c block peopl pass comment happen even
micro bu man hardli squeez breast hard burst tear
group boy comment tri take pitur
boy teas tri touch bodi
girl sent away home sleep corridor man pretend help took away home middl night start touch rape man also gave money
two men bike push snatch necklac
boy rub peni backsid
rape lang ata sent thank help
someon pelt stone room window face seclud street look man masturb street
confront person whistl walk past
went dinner around 10 pm saw girl follow comment indec guy
man call name made uncomfort
middl age man sit besid auto ws intent tri touch fake activ like throw packet tri pull plastic present side auto rain sunni
ithappen travel back home follow tri convers
face sexual harass public bu lot time recent teas touch privat area man feel unsaf public vehicl
get back market street corner notic 45 boy stand comment girl pass girl ignor walk past
obscen comment pass pervert
teas auto wallah exit select citi walk mall
man abus women road women
happen near prem gali near stationari shopin even around 6 month back go friend place boy stare inappropri
even
girl sexual assault mani day
hi bmtc mejast platform mani strang men roamimg time pass behavior embarrass women girl show reproduct organ women girl hardli tri touch ladi sensit part ladi hurri catch buse pleas consid big problem ladi
misbehav
girl take pictur anoth girl secretli saw girl ran away
ladi uncomfort two boy ogl rush
twelv year old electrician use come fix problem often came home come stay work alon use touch breast bodi
touch inappropri
touch
comment catcal whistl touch
around afternoon comment badli crack stupid joke delhi metro
9th std travel home alon public bu colaba peddar rd uniform sat last seat man 4045 year sit next open zip began shake show froze know theni thought tell ladi got casual went front seat save came home speechless told maid made feel better happen bu 8386 colaba opposit regal cinema
age 15 yearstouch tuition class 2015
morn class wait bu stand continu comment stare come way pretti ridicul
guy purpos bash grab slap
follow 15 year old boy skywalk vile parl
guy car stop front rikshaw follow us
gym work last set exercis everybodi left alon switch camera realli idea intent suddenli trainer came tri stuff two month earlier
bu boy push fall upon girl tri get close mostli men nonsens
man touch inappropri bu way goa
comment metro even
touch middl age man 40 year old got panick even realis somebodi clutch bag luckili money return home
k c collegechurch gate railway coloni back side colleg build everi men boy build harass colleg student girl stare comment facial express colleg professor take action
boy stand besid road cross road took pictur even pass lewd comment
travel metro group boy click pictur make girl realli uncomfort
comment happen afternoon
touch grope ghatkopar night
stare comment
tri touch hand bu stand
kid time guess 2nd standard boy name pradeep use play everyday took home start talk cloth said play game suddenli mom came took back home time knew exactli happen grew neg impact mind
inappropri touch behind streetlight effect dark
guy tri touch
four friend visit place go public bu middl age man push us backward told stand properli
old man cattch hand even told put hand away
boy ride bike comment us
go campu boy bu station want come told follow tri follow bu bu driver threw group foreign walk citi guy start follow us everywher end start ask whether ever sex etc
sexual abus boyfriend
guy stare continu wait train
sit auto wait friend arriv man appear nowher start ask friend know know u want friend tri touch auto driver stoop made go away later told drunk dint knew
travel auto sone distanc guy join two ladi auto guy took seat besid start touch inappropri auto start first scare realiz hell gave permiss touch rais voic told behav ladi also join asham act left auto
girl go home school boy start harrass touch bodi couldnt react caus alon
realli bad
man step feet even say sorri smile funnili assum
micro bu seat stand conductor micro bu touch back make uncomfort
guy send onlin messag photograph mani comment leg also part call ask meet matter report polic warn
chain snatch
group guy comment
street light mani part area
walk car desert park lot boy grab behind unspeak thing
hoy mientra caminaba rumbo casa con mil bolsa del sper un hombr se acerc hablarm seguirm por toda una cuadra
report submit safewatch
boy stand along street ask young girl go drink pub bill paid
 56% |########################################                                |
two boy whistl girl pass
follow
group men comment friend climb stair metro station
guy make appropri express
well sexual harass mostli common girl rare happen boy girl main victim sexual harass never harass friend harass level 200 student uidb buea come back school around 8pm happen street near hous
2 men tri grope sunday market start shout hand went crowd could see
friend sexual harass walk street man felt
sinc girl train honda plant everyon use stare
left hous work sat rickshaw notic guy smile look like belong villag nearbi initi didnt pay heed got sanpada stn saw smile look toward day chang time use wait night know got know exit time offic
street sunday even follow luckili shout ran away
rickshaw boy pass comment tri touch
way home guy follow ask make friendship start run start run behind scan whole build search
girl went help self forest stranger nowher tok knife ask scream rape
whenev friend goe market even two boy keep whistl pass comment bodi cloth happen near kamad 2 road street 8
3 peopl share rickshaw next iskon priest cloth show whole 5 min ride hand kept move thigh person got priest move away told took place mira rd shrishti complex
go work afternoon walk road near maharaja signal saw group mani men around 1015 men area seem like northern part india decent salwar kameez cover dupatta men group yell say quoti love youquot demean manner seem polit motiv elit white skin north indian consid god goddess wherea brown skin black skin south indian creep evil witch india white skin north indian women consid innoc angel heaven problem brown skin woman blemish skin call creep wonder real creep men catcal woman men pass sexual advanc woman 8 pm night educ black skin south indian women treat like quotnaukraanisquot elit north indian consid superior color skin wonder real creep men rape inflict harm women get rape wonder real creep peopl treat women like object sexual desir women courag step home educ fight grade call elit north indian give rapist us visa ask educ foreign countri courag instead belittl black skin south women made someth world
age 15 year oldtyp harassmenttak pictur agl leh 2016 28th march even catcallswhistl skalzangl leh time comment school leh time
two guy came huge bike snach neck chain could recogn number bike fell
accost ask give money tri grab wallet resist bogu policemen show took culprit away
incid took place even north campu kamla nagar road mani time boy comment cloth etc colleg girl
drug user complet mind masturb road
girl go coach boy click photograph ran away touch privat organ
alon mother walk holi udaipur holiday sever men approach us color even though insist want put color mother told leav alon afterward hug thought innoc true gestur happi celebr realis kept hand touch breast sinc decid ventur street holi
girl age 12yr play undugu ground boy ask take aunt given money told tell anyon went rape
sister friend wait auto outsid dwarka metro station final got follow group guy car start comment happen night
someon know victim eve teas harass
incid happen girl walk road boy form group bulli tri touch
way bu stop group boy comment
exposicin indecent del aparato reproductor de sexo masculino
return colleg boy stalk
touch inappropri boy
guy tri touch bu
saw girl call boy around 15 yr laugh went ahead follow boy darker space
poor street light katwara sarai result harass
board train two men pass across murmur someth humili hear
eve teas walk road
incid took place street connect e block f block naraina vihar afternoon 4 even around 7 person masturb invit sexual activ ran away order avoid
misbehavior boy girl pass vulgar comment
saw group boy tri touch girl n girl say n help n boy wear ran away frm ther
group boy bike hoot go friend
man masturb near park
sexual invit adher night hour
head hostel rickshaw strang guy startd take accompaniedd hostel ask number
guy stalk give sexual invit
indec comment
group boy togeth comment make feel insecur
go home tuition around 8 even guy follow around 3 month
girl touch odd place stand crowd place
touch breast break time
dim light lack proper sign lead emerg exit mansarovar station
taken famili friend go get someth famili father demand found somewher els room lock put key pocket rape mouth tie piec cloth
happen come back club danc street group boy came ask move hold knife stick ask give everyth walk away phone thousand
high pollut citi korba chhattisgarh
girl alway follow group boy go school guy follow got scare even tri touch follow share stori mother mother made complain polic station guy custodi gurl safe
dtc bu wit guy tri touch girl follow even step bu even
incid took place near vasudha apart even boy forc pull girl make sit bike
old man grab wrist start ask sexual favor get back home happen balaji chowk crowd area afternoon
drunk man touch brest
happen anytim normal happen everywher walk road
group boy use teas girl take pictur
boy men donot decenc toward women need walk street face kind harass
night phone stolen
happen afternoon
near hardrock cafe man tri grope
stalk alon
stand bu stop two guy bu stop crowd look around chang posit stood femal stand bu stop see face could recogn guy could cheap thing stand near ladi wait bu guy also came near toa like also chang posit like notic action saw give cheap facial express see chang direct face opposit direct also notic contin smile look talk tri judg board bu also board bu boy stand next told parent wont abl travel public transport sort thing happen
need specif date experi renew everi time go rathifil bu station near secunderabad railway station long area grope buse bu stop indec stare attempt stand close rub bodi take buse busstop 2 year stalk stare grope rub hope better light upkeep presenc policemen chang situat
wait outsid lilavati hospit around 1130 pm weekday man motorcycl slow front grope breast harshli drove away could anyth
man probabl slum snatch chain
friend rickshaw come home dinner foreign driver difficulti understand direct crowd men gather round presum help reach rickshaw grab friend breast angri jump vehicl began yell curs assail quit scare reaction ran mani similar situat happen time india go stand quietli anymor like asham
comment stalk
travel bhaktapur group boy touch uncomfort
go public toilet home harass month
mani time wit harass delhi rohini sector 9 13 morn afternoon even common topic discuss
man came near invit sex
walk street man suddenli grab breast ran away nowher found
vehicl go school got stage tout decid chang us onto anoth vehicl tri stop becauss get late driver insult quotyou bitch shut upquot
local touch inappropri 3 time crowd fled shout seem bother
visit templ boy turn turn push ran away
touch biker
afternoon two men came bike snatch chain ran away
someon took pictur without permiss
got train move staircas felt touch rear man behind twice
four guy open jeep made indec remark happen night
mother went shop mother busi look differ dress notic stranger stare follow went mother told everi thing told staff member person thrown
someon whistl chang rout
2 boy start comment loudli train
market buse guy nudg pass comment make feel uncomfort buse overpack andcrowd
professor track record sexual assault immatur gestur suddenli hug held made project student uncomfort also anoth incid scare anoth vacat student program student push behind
incid took place metro boy even age men stand women coach second coach stare girl girl alien seen girl first time
friend enter lift old man also present old man tri kiss friend happen anusudhan societi dwarka sector 6
 58% |##########################################                              |
dmrc even two guy pass bike whistl loudli
girl stuck even car broken guy approach pretend help happen worst nightmar could imagin
go shop buy groceri item even time boy follow comment know feel scare move away
guy touch friend inappropri even
ride scooter ride side big bu came wrong side driver laugh take danger thought die moment face type incid sever time ride scooter ride scooter safe kathmandu
afternoon go home guy whistl
guy tri touch inappropri long distanc passeng train mumbai kerala
friend told us incid school way stop guy held wrist start ask kiss
man came close friend bu touch privat part
around 5p colleg return back toward metri station came across drunk boy take pictur mine comment feel unsaf travel alon
pass coridor gatwekera man nowher stood touch breast sya good suck felt uncomfort embarass
misbehavior boy pass vulgar comment girl
man sing obscen song
rape seen 12 midnight girl found nake
watch tv cousin hous friend came suddenli cousin friend start touch got ran
even happen blue line metro
indec exposur
sexual invit take pictur
chain snatch
peopl tend take advantag crowd dtc buse behav indec
four girl walk street namast supermarket late even man stand road girl came nearer man show peni girl ran away due fear
middl age man tri put hand near girl privat part take cover bag
ogl facial express
stand outsid colleg wait auto boy came start stalk us
teas base color cast
travel crwode vehicl man pull unnecessarili
metro station guy came hit breast could react
sister stalk way back home sunday market
chain snatch outsid metro station auto driver
rickshaw friend two boy bike came near rickshaw tri snatch phone hand
class 10 boy senior use n say love lot dont like blackmail show differ emot lisent broke head injour total harresh
ride crowd car mumbai local train hand grab bottom turn bodi hope distanc anonym assail crowd move far time grope front upset nice coupl let squeez seat next get away man would stop grab
boy like spank girl abus pass way school lunch time
go class group guy stare
night comment
sector 10 market
two guy take advantag crowd snatch girl chain happen night
kamla nagar around 8 pm group boy look like colleg student splash beer dress ran away fast could told friend want report polic studi next 2 year
boy crazi pull cloth walk street
age 15 yearstyp harassmentfaci express leh market 2016 time catcal school street leh 2015 timeeven
saw guy comment girlfriend figur walk
afternoon full public view man stop flash wait sidewalk
person kept push behind public bu much space stand
three boy sing song pass
sing song whistl
stand outsid gym alon driver pick due unavoid circumst got late time continu chase group boy stare till time driver came
go school deep vihar boy approach took hand forc tri grope want come somehow push ran away happen earli morn mani peopl around
walk back home bunch guy whistl follow time
girl pass group guy pass comment appear
aunti rickshaw chain snatch 2 guy bike tri chase car unabl
girl abus policeman happen night
guy took pant happen morn
go back home two strang boy tri talk ask go home
happen eveninng
delhi rewari passang traincom
come back home coach class suddenli someon bike held hand increas acceler bike
gone shop near hous buy stuff even 2 boy start follow got scare shout came help walk faster spot father call boy ran away happen vika nagar street no2
teacher beat student
use go tuition near tuition boy alway use pass comment colour cloth use happen even around 3 month
misbehav
guy comment
group boy blow whistl comment outsid colleg
go walk everi morn lecher guy alway stalk stare ladi pass
misbehav
travel best bu man tri get close warn move away
went walk earli morn around north campu area auto rickshaw driver start masturb saw
momo chowmein type comment
place wine work done sonm peopl drink mani peopl mental disbehav
600pm friend go road side item return young man call ask know particular place ask found friend said ye way place hit friend back
peopl tri take pictur us also comment look
move around chandni chowk market friend mine wass pinch twice privat part decid leav market without purchas anyth feel extrem uncomfort incid
go colleg guy stare whistl comment follow
happen near power hous two motorcyclist pass give weird chaep comment whistl happen night
panjim market someon touch pass sinc crowd place
catcal
travel chandigarh delhi elderli person sit next complet half journey place bag seat squeez space seat tri sit closer tri put hand waist shift end seat avoid harrass
safe till date seen lot case like boy teas 4comment rough word girl
harass home famili member
even gate peopl alway comment touch
man show privat part
take pictur without consent jammu
teacher beat pupil
grope disembark train dadar station gener compart man held breast press lead shock got shout tri catch mingl crowd kept stare shout abus could run due crowd creep walk
forc kidnap rape gunpoint 2 men move car near dp rk puram
presenc gang male stare femal pass comment case emerg mark escap women
pass alway make signal among start call pretti like
stand near ticket counter metro station wee guy continu stare make uncomfort
guy lust
sister friend stalk three guy continu follow almost 1520 minut
sat car group boy start make facial express tri talk
sister follow unknown person bike
went bu bu got crowd guy tri touch feel night
went khanpur shop even sister group boy start follow us start walk fast also chang rout mani time would stop chase kept follow us went insid shop stay long time women shop ask us happen told also told boy still outsid shop ladi went chase boy away month
cousin friend juic even lot rush 2 guy supposedli stare sinc crowd werent sure start walk back pg met anoth friend talk told us someon stare turn saw guy realiz stalk kept take differ rout lost went back
men darajani call girl pass tell want someth import tell girl also promiss girl money
catcal comment bad facial express
case common area though never face heard lot
yesterday way sankhu girl two boy wear colleg dress wait bu boy talk use mani rough word girl feel odd scold boy
two boy teas girl
walk school came acroo group men stand outsid vehicl said big butt
harass
rob dark street corner 2 unknown men
know happen girl women face everyday men walk toward bump basic tri kinda physic touch say anyth blame place crowd never meant anyth obvious girl come know someth done purpos happen mistak hurri stop everi 5 step slap man tri ye frequent happen everi 5 step crash man walk toward
travel girl friend munnar bu halt somewher munnar 10 min break roam take pictur suddenli man expos dk start wave toward us embarrass move ahead month date locat approxim
touch grope dtc bu near munirka
 59% |###########################################                             |
happen friend get back shop men approach start misbehav comment ask sexual favor happen rajhan vihar afternoon
boy comment figur
go mosqu men start call shout tell quotuko sawaquotmean cute
man touch inappropri bu
scooter red red light micro bu conductor touch upper bodi part pass spontan could even react
stare unnecesarili comment wierd stuff
catcal whistl comment ogl facial express
follow
poor light bandra station dark unsaf night
walk olymp road men idl outsid movi shop stare girl pass pass well pstare
two guy tri block path tri touch
sister go market two chain snatcher came bike snatch gold chain
harass midday
villag hardli get electr whole night would electr villag forget street light villag remain total dark night unsaf girl
man follow till hous
ogl comment
matern uncl home friend tri harass touch unnecessarili even click pictur
misbehav
boy sodomis father
woman rape rober go work earli
sexual harres ladi travel micro bu
way back church xma eve girl saw walk saidquotwil like sex mequot repli got angri friend attempt remov trouser forc sex thank god brother came rescu
realli bad
man follow touch behind
catcal comment bad facial express
ogl metro buse follow stalk till home stalk phone
girl pull men
evenig time 3 boy tri touch aiim bu stop
near park follow boy even around 6 month
guy follow car colleg chanakyapuri
sat auto go malviya nagar man came nowher touch inappropri shock happen near auto stand dakshinpuri
age 16 yearstyp harass staringogl choglamsar leh near choglamsar bridg 25th decemb 2015 even
cheap comment stranger
come across fool vagabond almost everi place colleg life pass indec comment whistl indec manner show vulgar gestur embarrass person girls0
walk janakpuri metro station aakash institu small littl children tri pick pocket
travel somebodi stare follow
come home offic goonda sit nearbi comment appear
sexual invit hoot comment
2 guy bike stop car took key enter car snatch bag fled
walk street last even biker zoom past yell quotwhat babyquot obvious could react point happen fast speed
20yearold man allegedli tri abduct molest 12yearold girl safdarjung enclav return school wednesday afternoon
chain snatch theft
pass friend place man look homeless start stare follow distanc moment realiz follow took rickshaw quickli
saw girl conner boy touch privat part trie stop slap told get
misbehav
friend gone market even guy approach told like want friend told go away made sexual comment girl told mother turn blame wear right cloth go
boy keep misbehav colleg told complain continu scare report
month back maid gone usual dryclean dri cleaner inappropri ask join hous insist come upstair anger maid told behav complain
young guy show privat part stand next girl afternoon hour
comment amp take pictur
misbehav girl go buy cloth
gone buy alcohol local shop 2 men pass obscen comment say characterless woman drink
realli bad
follow stare cross road go nearbi market opp ananda apart
happen near school
follow lane commun
go home colleg rickshaw guy car side road whistl pass comment
vaishali buy fish guy start whistl catcal
even group 4 come practic phone ea lost way
guy shout hoot girl pass comment give uncomfort look
morn comment wrong word group girl
2 men comment cloth bodi
head toward home peopl follow scari
unwant comment ask seat guy
happen went kerala travel bu look window saw guy stare make absurd face
friend guy continu whistl comment wait bu
auto driver kept drive absurd place
gone shop around newroad area men came near behav weird blink eye teas show bad facial express
men touch drunkard ladi aimllesli take advantag condit
girl allow get car shout driver drove fast ran away forc remain car happen near mini seashor sector 10 vashi
peopl comment dress stuff kamla nagar area happen even
realli bad
person touch hand bu travel dolakha district
go share auto station r citi mall ghatkopar mumbai boy sit near take undu advantag
chain snatch morn
man bar drink call pass ladi enter bar window reach near call ask himquotdo know mequot said quotcom bottlequot told quoti type excusequot went
walk road nearbi shop mall man pass say bad word privat part
night sent mum buy stuff met guy tri convinc intervent firend help like would victim
uttam nagar termin bad boy start comment girl pass area
older man stare spars popul bu stop busi nandidurg road around 3 pm jan12013
friend wet rain boy comment us
bandra fair group friend boy girl guy pinch ass realli hard time turn behind lost among crowd still idea
ogl stalk
teacher beat student
travel bu pepsi cola ratna park boy say bad word amp gestur annoy
walk street boy bike comment
harass rohini
stalk
wrongli behav
9year old school girl shadrack kimalel found injur bleed behind lindi friend church lindi villag defil left scene
man walk 5 minut hand insid pant tri show dick assamp ran life till saw shop call someon come pick
report taken dadar street market exact locat bigg salemaitreen shop interviewee feel safe due instanc stare touch time visit
male grope back travel gener compart friend
way home pass olymp stage man winkl avoid walk toward ask number claim beauti girl walk fast never saw
wait train man would stop stare
follow call messag difer contact number
matatu two young girl olymp makina girl alight karanja stage tout convinc go town talk also touch
 60% |###########################################                             |
friend walk lane man call name teas
tri touch hair comment beauti
chain snatch
friend mine go walk near northern ridg area somebodi start follow felt uncomfort start run escap
follow written horrif experi 23 year back quotteri chut lene maza aa jaegaquot said mayb fault never known mean repli quotmar ja jakequot voic quiver everi sharp exchang word later quit time thought better respons could given quotpolic ke pass chalega mujh bhi maza aa jaega quot quotquot well thought still abl put word suddenli given glimps would feel like rape victim probabl would clichd say suddenli felt expos everyon around could see indiffer mani eye around knew happen middl road broad daylight pleas note still speak like rape victim saw mother daughter probabl 810 year old gap 1m made feel insecur want shout mother grasp daughter hand hold close suddenli understood mother protect knew still somewher around watch love get high reaction caus thought made lighten didnt want give satisfact look around cheerili well realli normal enough men around suddenli seem leer casual glanc felt like pervert gaze wish pickup alreadi stupid 20 someth year old encount innumer time suddenli began seem threaten judg know sex addict 1st hand experi lust addict peopl around suddenli thought come gut mean stage addict im assum blunder attract say kind thing young girl probabl compuls must start somewher dont know singl person would say loud someth like even relationship naughti role play night believ tougher propos girl mayb isnt mayb brought famili angri father told daughter heavi abus mother openli told father pleasur tonight sibl would call swear word involv mother sister casual call prostitut told thing wive fit ps pleas note date n time approxim
two men stare woman
way colleg travel bu pune man make facial express toward tri hold hand confus walk away next stop mine
girl come school group boy teas say bad word n sentenc could take action ignor
chain snatch
near templ commun indira nagar stalk
ill treat bu
experienc whistl teas street around home area
crowd guy kept touch tri move away kept chang place next
ogl grope
case like robberi theft
month ago lane bhatbhateni found guy whistl show bad facial express
way ghatkopar dn nagar metro station saw man assum must marri ogl women platform gave man stern look whenev saw pretend look somewher els noth
boy gang around start comment
man thin voic yell somewhat low voic bayot becam appar wasnt respond step onto sidewalk wait walk toward say thing tone voic bayot
person comment pass dirti comment even hour
way back home colleg saw middl age guy comment follow ladi could much help awar relationship guy look rough
auto wala stare make express inappropri
colleg student stalk comment girl fest
touch grope surat got annoy ignor thereaft
walk shop man whistl call
happen peeragarhi bu stand around 400pm wait bu group 45 gulligon whislt everi possibl girl huge group behav indec
happen almost time
saw old man hold class six pupil 5 00pm
girl beaten men later took hous rape left dead
sartuday morn friend appoint catch guy sudden taxi enter rascal took bush badli rape until fell unconsci taken hospit given energi next minut pass away
comment loudli bu stop front everyon nobodi anyth stop asham react
outsid comment girl
man came closer walk way start make differ facial express
harras afternoon
way colleg boy start whistl comment
realli bad
travel place rickshaw guy santro car follow stare
go market alon boy came bike took girl photo teas later ran away
girl around 1920yr wear short contin stare comment upon group guy around 34 awkward other happen afternoon blue line metro
uncl middl age pinch breast
wrong touch given teacher
sister friend stalk three guy continu follow almost 1520 min
reckless drive market area
indec comment afternoon
guy yell physic assault girl road 1 00 p go downstair interven stop told would report stop horrid peopl watch happen stop man yell home second floor would call polic kind apathi quit scari
girl go attend parti even time cross narrow road way 4 guy stare teas use bad word afraid go home
come back colleg boy follow road
small girl age 7 year rape old man 57 year earli morn mother left hous
comment walk
guy said friend hey beauti pass
bu person sit next continu stare seat chang start click pictur
person constantli rub elbow bodi sit besid auto
man physic abus touch breast insid bu
saw man snatch chain
inappropri touch
show privat part guy travel afternoon
2 men comment take pictur mall
group men made obscen gestur pass car man metro took pictur
misbeha
govt offici time feel unsaf due overcrowd area
insid class room peep throughbth window saw man live theschool compoung whink make suggest sign
come alon 9 boy comment whistl
harass mi behav
bu conductor rub bodi mine go around ticket
pass men start talk stare
dadar bridg incid grope touch crowd miscreant tend take advantag
eve teas late even road desert
bu wear short man put hand thigh took hand sometim thing repeat sever time chang seat
went shop sarojni nagar boy make facial express scold badli
guy whistkl
chain snatch
pass railway line friend man pass us stare eye eye turn back still look till fear
saw group guy take pictur girl sit pavement nearbi
man deliber touch femal copasseng dtc bu 957 bu overcrowd
train delhi rewaricom
happen afternoon
comment made travel afternoon hour
auto driver follow kept make indec remark sinc mani peopl around sure react quickli went bylan would follow
grop boy comment
man approach anoth direct said someth got clear headphon hear said felt malign ignor kept walk got stuck traffic light turn around still make lude gestur stalk
man deceiv young ladi want send shop girl went hous man close door
return market boy start follow gave vulgar comment
took pictur amp comment
sister return work kandivali station platform 2 old man get stair make obscen gestur face confront quotold man age thisquot took slipper hit everyon look
hous 1050 pm man walk hous knife close door home friend pregnant littl sister boy sexual harass us knife made us understand friend outsid
group boy stalk friend
return home guy came bike start comment react ask tell mom sister
guy use harrass back school incid made psycholog mental disturb
head stage board bu someon grab hand ask number tri remov hand held tightli everyon stare us felt asham let go hand walk quickli board bu feel embarass
touch grope
friend stalk guy amp got someon amp start messag inappropri
girl come school shopkeep call went shop man took girl
ogl facial express
walk street
touch whistl follow
car 2 men follow car tri deceiv
cab follow throughout ask sexual favour
guy whistl girl wait delhi metro feeder bu servic janakpuri
enter matatuand man held shoulder remov hand click wink
swim teacher tri assault
go market boy came bike tri touch privat part
chain snatch happen aunti near sharma apart dwarka sec 10
walk toward station peopl stand outsid metro station made lewd commen
lot harass take place near area girl harass peopl roam around knive
bu phone stolen
classmat told move even neighbour buttock
n colleg fren return home guy came talk us told us interest stuff show us unknown person insist us went narrow subway overther show us pron site
walk man opposit side grab breast ran away could anyth
 62% |#############################################                           |
make facial express amp stare continu awkward way also face physic harass
friend hit shoulder full forc guy motor bile miss balanc fell broke mobil
best friend molest local train
elder sister walk street guy follow us scare start walk briskli whistl also start walk briskli hurriedli went crowd hide slowli crowd stop safe
return home metro station two boy bike touch behind could anyth sped away
forc
chain snatch
touch ogl
old man pass young girl tri touch girl
abus word
incid took place morn sector 13 rohini walk colleg pg boy group comment us
daytim way kapan bu man age around 30 next bu crowd person intent bad sat side start touch
lot harass take place near area girl harass peopl roam around knive
boy age 15 year attempt theft bike due live monitor cctv boy captur send polic custodi
girl short skirt stand bu stop men start call name make run
6 30 walk toward bandra station complet dark seem road suddenli someon caught back touch near crotch turn saw man scream fright shout abus ran opposit direct time peopl came station toward alreadi gone cross 60 year age extrem frighten traumat experi
go sister hous man gave lift talk nonsens later gave visit card give miss call would call back even said would take
walk way station man surreptiti came behind touch butt fled
go shahdara metro station way man start whistl pass cheap comment metro peopl touch pretend noth happen
way colleg soon enter station heard boy comment dress style
drunk guy tri harass move local train
got annoy ignor thereaft
teacher alway harass
follow comment
go home offic offic hour alreadi late even catch cab instead cab driver could get eye felt kinda offens
pass corridor near build also man pass stop walk thought give way reach near open arm wide hug refus abus say stupid idiot
harass north east indian
happen street live 14 man much older kept stare disgust way
catcal comment bad facial express
boy dont know touch tri drag away fought back
touch bodi part girl push incid took place public bu shout necessari speak
boy comment stalk girl
friend got whistl foul languag group local resid guy
men tri yo touch girl
go auto suddenli car 56 guy put realli loud music disgust cheap song make weird facial express follow us around 1520 min realli scari
friend harass someon pass comment moolchand road afternoon month
feel safe station
catcal comment even
whistl
walk street sister boy teas
murder
daughter play outsid neighbour gave sweet took hous look found neighbour bed nake
accord unsaf place delhi sultanpuri mobil snatch beep stare taut girl
friend kiss street guy ran away
pass kamla nagar market even around 6 man 40yr age pass pass vulgar cheap comment embarrass felt kind eyerap
go school saw group boy start whistl repli start insult
man touch wrong way
stay bang smoker smoke way toilet whenev pass tell take puff never given go help
way toi market man start insult say big boom got shock sinc understand said embarass
go work way colleg peopl use pass comment like chines chinki etc
pictur taken comment pass
happen sister use go colleg earli morn shown sexual organ done unsaf sexual behavior
chain purs snatch
case acid attact girl reject boyfriend due person problem girl walk anoth boy boyfriend attack acid
bu
walk road morn toward lalit man car come opposit direct slow pull window yell quotmast lag rahi haiquot quotlook goodquot vulgar manner
took taxi borivali station east thane reach ask night fare even time 5am ask 70 rs toll although paid toll use name avoid toll import gave 2 note 500 quickli exchang 100 rs note said provid 600 inr read incid mani time knew fraud check found claim correct intervent secur guard taxi driver went away bad note taxi number maruti van
attack rob gold
went meet senior person respect uncl restaur time none tri kiss touch privat part could react public place respect felt fast
harass afternoon
bu go colleg man start ruf crotch shoulder
friend touch privat part man bu travel
man comment women stade local train man commit local train
experi bu boy touch bodi disgust
men whistl follow girl wear short dress
ulhasnagar around 10 pm return home finish work walk toward road found mani drunk men dim bridg sleep stare creepi
way wed girlfriend wore decent cloth bu stop men tri remov cloth beat defend reali touch privat part
touch privat intension
chain snatch
harass morn
happen late even travel subject variou catcal whistl comment happen metro dingi quotgallisquot citi
take pictur without permiss even tri touch
815 pm come back colleg suddenli van stop besid start pass comment gear pace rush insid colleg gate
woman touch inappropri man sit next bu react thrown bu sent polic station
chain snatch
incid took place afternoon metro blue line guy compart next women compart stare comment
report taken dadar street market interviewe feel safe due instanc push market
rickshaw suddenli two boy start follow bike whistl pass cheap comment
walk kianda men idl outsid shop call girl pass
old age man touch hip bu
walk back jangpura metro station policeman call protector common peopl pass cheap comment stare eve teas common girl experi everyday life
wit incid chain snatch morn hour
two young boy along age man badli comment two girl
grope pull dupatta
guy make differ facial express univers campu delhi afternoon
survey carri safec red dot foundat along safec audit street market mumbai
guy contin stalk stare women pass obscen comment
abus harass came douala first time school first soon student discov bamenda took mockeri time class stood answer question would alway sayquot sit bamendaquot alway walk around class alway mock say quotca ce en bamendaquot peopl someth person bamenda like sit quiet bamenda peopl know anyth
way home drunk man push back
travel bu aunt two unknown person touch us unnecessarili knew feel uncomfort still give
common thing
stalk stranger go coach
return home micro bu full time middl age man touch thigh
chain snatch touch grope
sing song come school said ok happen near hous
ogl facial express comment
step build guy whisper someth realis pass stop path turn around look back reaction let ruin
receiv vulgar comment travel metro
go outsid visit place boy commmenton us abt appreenc dress also whistl evn group boy comment lot useless thing n folleow boy stare felt uncomfort
teacher josphat touch anoth girl privat part
peopl offer lift way colleg even deni use follow
kali templ lucknow cop took advantag crowd kept touch girl less 13 yr old inappropri due crowd realis immedi soon realis continu press breast
crowd local train man kept touch throughout journey got last station whisper comment ear went away
take pictur amp comment
class eight girl bought sweet told remov cloth man bought sweet refus man rape
due poor light abl catch peopl tri grope butt
person travel share auto tri touch
 63% |#############################################                           |
travel dtc bu 253 use travel everyday boy board bu brj puri bu stop start stare contin stare irrit got irrit decid also start stare soon start stare give tough look becam consciou got bu
got toi market makina stage way man accident knock forehead use elbow sorri said fine follow held waist still tell sorri told ok iremov hand waist held waist hold tighter struggl pull walk away quickli
friend harass boss reput compani kiss forc without
happen famili step father abus physic year
girl deepli depress boy school use teas obes person could stand fact tri commit suicid counsel well
travel girl friend bu halt somewher min break roam take pictur suddenli man expos bodi start wave toward us embarrass move ahead month date locat approxim
n sister return home boy start teas us littl drunk n smoke ciggrat 510 boy start take pictur 2 follow us escap run fastli
continu sexual invit travel mostli happen afternoon
snatch chain near mandir
man ask possibl go told time
alon home religi teacher show crotch
morn druggi start touch inappropri made uncomfort
guy touch girl back make use crowd platform train arriv happen thane station platform
stalk villag 27th februari 2016 time
girl harass famili didnt support friend made complaint nearest polic station
group men stand comment everi girl pass
way go meet women dc alight bu stop groupm men seat road side invit ladi go nad greet ladi ignor start insult
vulgar facial express
stalk comment
househelp use sodomis form 3 boy whose parent work away home whenev everyon hous boy would come back hous sex hous help yet parent thought school took intervent teacher ask parent nolong come school concern neighbour househelp reveal save situat
friend walk school along darajani group men sit pole start call us mock dress mode laugh felt reallybad
public vehicl man grab finger scari chang seat
comment touching0grop amp bad facial express dtc bu 569
survey carri safec red dot foundat along safeti audit street market mumbai
afternoon 2 30 pm
stand line metro got attack boy tri push touch
new place shift go find good near shop teas comment druggist scare call brother warn druggist
walk back home guy come toward bicycl touch breast went away scare embarrass time
school carniv men tri spank even though teacher around femal stand male friend stood almost got beaten made sure men left campu
guy teas comment girl
touch
rob gunpoint three peopl
friend go anoth friend home
teenag girl go toward metro station car caam near boy sit insid open window gaze toward ackward manner said quotlet goquot
girl walk hair clutch two boy came bike pull clutch back
group boy threw flower comment indec walk ahead
public vehicl baneshwor fren guy tri touch hand tri avoid first keep tri touch could speak ignor chang place felt embarrass
suffer common cold went clinic way doctor uncl treat unusu
man took advantag conjest start touch ladi could avoid due multitud
went afternoon boy comment happen afternoon report polic respons
harass
two guy tri kiss molest parti
friend work oil compani biratnagar live alon work 8 7 pm return boy follow till room feel insecur walk alon particular way chang way room
alon home man came place mother domest help hous inquir mother told home come back even told meet start touch wrongli start cri went away mother came back home told told taken 10 000 rupe man came home even start threaten mom return money get marri mother without think got marri rape everi night scare anyth
survey carri safec red dot foundat along safec audit street market mumbai
2 men comment walk
local bu boy tri touch belli shout
girl kiss guy gunpoint front three policemen anyth happen even main road
friend group walk road boy came speedi bike tri snatch chain friend pull away
incid took place night main market four friend two differ car ofth friend shot friend sit car polic alreadi
go colleg andheri east group boy sit whistl
year 2014 came across incid young girl 17 year old rape abandon near bush reason drank boy bear promis date boy lateron
ogl weird facial express
incid happen even roam friend
walk small lane cross two street man bike came behind touch back sped away within second could notic bike number shave head wear cap
man grope breast enter train
morn went shop mother call filthi name guy
women harass catcal whistl tirumurthi nagar nungambakkam chennai tamil nadu india
comment catcal chainsnatch
harass take place everywher delhi safe happen even
comment amp take pictur girl came offic
seen ladi cri chain snatch incid happen jj coloni uttam nagar
incid took place less 1 month ago even metro station person metro attempt touch privat part behav accident hit
even indec exposur
take ticket counter line boy tri grope butt
realli bad
got molest random guy
man stare make uncomfort 14 year old
boy follow
peopl made vulgar remark
open wire hiranandani kensington wing near octaviu lobbi realli danger anyon
chain snatch even uttam nagar east metro station
proper rule regul bulli rag becom victim sexual harrais stude grade 10
central secretariat see lot crowd board board metro friend deboard felt somebodi tri touch lower back
sunday go church boy came close touch buttock shout asham ran away
market famili man actual grope breast
incid took place night around 830pm outram lane near kingsway camp friend return activ class scooti group boy car make video soon friend notic took turn street
toler
catcal whistl take pictur comment indec activ
auto guy see alon auto took wrong rout say shortcut later shout said call polic took right way
get back home sunday market saw girl harass men whistl pass lewd comment approach girl told speak said everyday affair learn ignor especi women
group boy comment even stare colleg canteen area
block way could go car
comment
old men around darajani area like give school girl money way school
share auto last stop guy left suddeni start ask person question said kiss could react ran away
guy tri touch also comment
man tri touch thigh turn around slap
comment catcal whistl
buspark man follow throught balaju rought alwa catch hand told touch said want howev went parent tell incid parent react punish
chain snatch night
man sex daughter sinc tender age 16 year man caught 26th decemb 2014 sex daughter upon enquiri girl age rang 1416 year confess go 6 year old
physic abus bu educ person even time
come back tuition boy tri touch bda manner
usual even friend guy rowdi come along road start comment girl abus
return tution class street boy start follow ran stoop chase way home
incid took place rajiv chowk metro station even mostli happen time metro reach station men misus movement touch privat part
still villag electr ex patrapada bhubhneshwar pleas take appropri action first rather work digit india
call quotchamak challoquot afternoon
pass men start call tell im beauti
saw man teas tri close ladi insid public bu
leav guy stalk till took auto left place
grope 45 boy money
friend go colleg ogl men make eye contact
boy public bu gave bad look
 65% |###############################################                         |
comment
teacher told us prayer bring 50 peopl per student parti boy class told shake
woman rape laini saba never taken hospit
return home school follow boy made scare
man fallow pass comment
touch man insid bu
harass
stand outsid colleg bunch cheapster sit front us whistl comment
happen friend moharram near mosqu man much older tri touch inappropri
morn 10
given sexual remark
tri touch breast
hyatt mohammadpur subway safe
age 19 yearstyp harassmentstaringogl choglamsar leh time stalk leh main bazaar time catcallswhistl hous coloni leh time sexual invit ye even
jhangirpuri metro station saw boy take pictur girl without permiss
saw group boy start follow old ladi worn lot ornament show knife snatch ornament ran away
bu often
notic old man follow near hous
walk convent junction guy suddenli walk pass said quoti wanna fuck youquot sing song manner subsequ walk past without stop occur almost 57 year back approxim 5 pm date time given exact
way colleg boy back comment teas throw stone shout laugh stop activ
sinc new citi nearli idea rout auto driver misl wrong place
man happen pass girl walk area deliber touch hand pass close
iam alway haraas father
harass
comment
bu go toward sec 18
comment ogl touch kashmer gate metro station even
stalk
tri get auto post shop guy click pictur
continu stare tri come close touch
touch happen time market alway crowd could identifi grabber quothard tellquot made purpos knew never accid
disturb boy comment
overcrowd buse chennai male tri grab embarrass stuff femal put normal thing due languag constraint pretend understand confront stand public transport buse chennai
physic abus insid public vehicl
man came upfront hug tight felt breast
go catch train way home saw men smile toward whistl
touch stalk come back
group boy whistl said bad word
day ago walk alon road guy start lookin sing song whistl
stalk boy regularli use go school morn wait bu stop
travel train thane nerul guy start touch tri come intim zone start touch person part tri stop left
guy use bad comment gestur friend return colleg
happen walk road sunday afternoon peopl road sudden two guy mask came bike snatch chain ladi 40 walk km away ladi hurt taken hospit
bu old man stand behind tri rub
mob thug drugdeal stalk women catcal bypass polic noth polic ever read guy call mehdi tall scar forehead friend drug dealer
boy wistl comment go hous
go home offic alreadi late even took taxi taxi driver could get eye felt much offend
happen bu come colleg
cbd sunday two friend biraci descent nearli everi man pass us say someth like hi sister go look beauti especi direct friend also felt uncomfort touchyfe peopl toward especi maasai market cbd use stare never felt uncomfort pass nearli everi singl man street reliev thank get cab go home felt unsaf know friend
wait tempoo friend man came stop bike open pant chain jacket show privat part afraid move place later
shock middl age man grope crowd train somehow shout left
gave seat ladi guy behind rub
visit uncl holiday uncl remov cloth couch ask wa scare ran away
bu stand boy stare young ladi made feel awkward
friend travel nepal yatayat bu man put hand friend lap normal
gone mother dairi nearbi guy start follow realli scare kept pretend notic
travel bu boy comment took pictur
happen outsid ramja colleg afternoon guy tri copi friend tell someth made face well
misbehav
man show porn clip school girl afterward turn sex without girl permiss
friend touch inappropri place cousin gone visit
wait friend near colleg
harass even
boyfriend slap public space tri forc came help
guy came snatch aunti chain afternoon hour report polic
never harass experienc sever harass case someon know friend mine harass street rough guy harass slap friend good reason angri want take violenc hand
men whistl go home
bandra station man kept tri get close succeed touch chest
realli bad
group young boy pass comment
comment shopkeep
lot girl pg sector 14 guy white scooti might b activa spec tri touch n comment girl 45 time incid happend
touch month back
happen school right insid class lectur room 9pm read alon hall tall man dark skin guy enter came toward demand direct process touch breast knew press ground romanc carasss hip
incid took place even stand friend group coupl boy teas us
walk road man walk past suddenli squeez nippl never felt violat disgust life apart daili sexual harass whistl catcal touch squeez backsid
go shop girl also come toward shop left shop saw stand old man convers heard man tri go wish hear late rape
friend troubl guy kept misbehav sexual way make uncomfort happen sanjay camp gali 1
go toward home group boy stand side metro station comment everi girl pass
iv e experienc bad facial express shown toward
around 30 day back go tuition madangir c block peopl pass comment happen even
compound 2 strang men came claim knew father welcom hous next thing nake
girl sit boy tri touch gave express slap
girl stand bu station guy tri get close touch even pass remark girl rebuk guy stop behav way happen afternoon
went eye hospit younger brother way hospit peopl teas us sing teas
sexual molest hous
friend gone colleg fest program go variou guy came surround group danc floor start pass comment grope touch
conductor bu group girl travel follow us till krishna templ patan
follw home guy bike
old man misbehav hospit touch breast scare anyth
harass even
comment pass foreign tourist visit select citywalk mall
titan showroom look watch collect soon notic 34 staff member stare also start stare time stop stare went els
2 guy bike snatch aunt chain walk home work
afternoon friend stare continu
afternoon riskshaw rel two bike came snatch gold chain
happen friend come back school certain boy approach said quotbabi youquot said fine minut later boy start touch
travel train man tri touch girl
go nearbi shop purchas item two men street start pass cheap comment gave dirti look happen somewher april
roam gip noida comment upon
friend beaten belt uncl happi last exam mark
steal comment grope
follow malabar hill marin drive
 66% |################################################                        |
guy continu stare afternoon hour
poor street light
boy near home give facial express
man bike indec expos front girlfriend happen 2011 morn report polic beat
particip crisscross citi road late night hour found mani stretch road includ arteri road like j road mani road streetlight either dysfunct dim fail illumin road pavement poorli lit mumbaibangalor bypass infam road mishap
auto riksha driver amp friend abus girl street
went shop store unknown girl ask cell number
way home local guy teas tea shop
wait bu bu stop man came near said spend night even follow 30 minut
touch sensibl area bodi
boy taxi adjac mine turn start pass lewd comment laugh
go grant road bu bu stop felt touch felt scare left bu fast could
happen dtc bu man came sat next kept come close constantli kept push hand away kept come close say crowd side get push
touch also facial express
conductor stare often inquir
purs gold jewelri snatch
come back class man scooti came besid show peni look toward move away scooti open zip show erect peni
comment catcal whistl touch
old ladi pull bush man went eas grope breast bit also rescu shopkeep nearbi peopl toilet n hous
man tri grope someon came 5 min ran fast could
comment pass trimurti seva sangh chawl
drunk peopl pass comment friend girl walk tri approach
harass
guy enjoy pass comment women look autorickshaw even
went drop applic told go boss hous grab serious start harass final ran escap
walk go home school came acroo em sit road state lok call talkingof look didnt like
girl live haider pur villag discomfort man constantli kept stare look cheap way
catcal comment ogl outsid pragati maidan metro station even
friend come school start rain run find place shelter found old man told us go lie
girl friend deni job refus secuum boss sexual demand
work factori bengaluru rural area street lead factori work street light becom extrem difficult go work night shift sometim even robberi also take place stranger group peopl came onto mug took phone wallet
boy stranger friend touch friend use bad word happen even
girl studi grade 9 sexual assault teacher class govern school jhunkhunwa rautahat
travel offic home late night journalist two men start follow ride bike follow till reach home comment indec manner
boy stalk week stretch scare anyth
car constantli follow made kiss face
group boy car behind us tri make video us walk
happen even
16 year old girl taken room man threaten knife taken isol area late night rape man repeatedli god grace found way home quarter pass
gangrap victim uttar pradesh ambedkar nagar violat investig offic io probe case alleg also rape inspector charg polic station lodg complaint
ride scooti three boy three differ bike kind teas tri come near
survey carri safec red dot foundat along safeti audit street market mumbai
morn guy tri snatch phone
guy repeatedli call harass told manytim call follow listen
per guess man nearli 3035 push left corner tri touch
go pick littl cousin school man roast maiz catcal whistl look call smill confront sinc hurri ignor walk away
return school boy comment tri touch
come back tution two friend n two guy came whistl back
walk along olymp road pass boy wer sit door sell cd higher ruturn watch by sat lookig comment girl pass among girl comment
realli terribl
2 friend went hangout mateo 2 men stalk us auto stand till entranc cafe
victim small kid play outdoor coloni gate unguard thu unrestrict trespass come comment took place around 7 pm even
happen near preet vihar metro station friend walk three boy bike came close hit us back
good friend mine follow metro station pg malkaganj guy riksha confront scare
area realli badli monitor area 90 percent rural popul crime often area
thief forc snatch mine chain ring lone place
9th std recess time school boy touch near stomach complain princip took strict action
vo bandi vaha k flyover k nich se ja rahi thi evng tym pe n tabhi phich se kisi band ne us pe attack kiya n vo zor se chilayi luckili vaha street k kuch log ne sunna n vo ane lage tab tak vo ja chuka tha
happen bu near munirka guy state comment dress code
friend ask particip poem competit competit went particip touch unnecessarili teacher
white collar men pass grope chest walk away
guy comment whistl girl dress short happen even
10 yr old cousin sleep room 13 yr old woke saw touch everywher time child understand
comment amp ogl facial express dtc bu badarpur 30th septemb 2013
comment bodi night
crowd bu saw man pinch unknown ladi
girl way walk guy teas whistl happen even
friend mine ask accompani hous reach close door threaten kill scream rape
friend mine rape boyfriend best friend went advic problem boyfriend incid occur room fought hour could fight
walk male friend knew sinc long time back touch privat part told listen came home leav
man follow whistl said bad word
realli bad
public transport delhi safe men ogl grope comment
walk road heard call ignor
survey carri safec red dot foundat along safec audit street market mumbai
wallet stolen
go colleg local bu man sit besid tri touch breast
harass
sing loud cheap song man found give sexual invit us mid night
happen trade fair even guy pinch realli embarrass crowd place
touch privat part intent crowd bu
cross road teas follow man also tri touch
friend sister come offic said rough look man sit next woman masturb bu bu light time man touch women said touch fear voic
reckless drive
actual big matter us ladi al way feel abus simpli travel public vehicl like touch push regular problem
colleg student boy approach propos ignor start send sm give blank call cell phone went home came surpris see stand outsid build ignor follow colleg also past 3 day realli scare throw acid pleas help tell thing parent want get unnecessari hassl
poor street light work red light
boy invit sex fb
get back home work group men start pass horribl comment charact cloth know react marri woman said anyth peopl societi would blame happen near kali mata mandir
repeatedli rape someon meant trust
someon know ride bicycl suddenli bike pass along touch inappriopr amp sat bike drove could anyth
ogl touch grope
age 16 yearstyp harassmentstalk leh market return home school even catcal chocho lay hey doll time comment moti fat balu small timeeven touch brother wed night
happen everyday
unknown man seat next group girl row f theatr nbr 2 watch newli releas minion movi shortli interv man took pant flash peni innoc girl sat defec right horrifi sicken complain cinema author took man task
chain snatch near town hall
comment dhaula kuan bu stop even
catcal whistl comment ogl facial express take pictur touch grope sexual invit
whistl stare happen even 5 pm
happen afternoon
comment
comment ogl catcal night
woman taken away hous group boy taken unknown destin gang rape
felt uncomfort boy way school usual comment whistl
happen grade 10 earli morn around 6am bid mother left school near home 40 year old man face toward bush got near start make wire sound glanc short view saw hold peni shake afraid call mom told help go school
night go back home two guy drunk came toward start touch help male friend
 67% |################################################                        |
guy touch friend leg wait
go shop man chew khat got hold hand forc tri grab fortun woman warn
ogl
happen quit number time walk friend even 23 boy bike whistl us
area govandi toward deonar municip coloni west near 600 tenement build evacu dark night time unsaf women girl come home offic even also street govandi staion west toward tata nagar alway occupi hawker everybodi feel unsaf walk road auto rickshaw drive rashli amp 51 peopl front polic man nobodi bother safeti area
boy follow us make loud nois disturb us even follow us
go school boy follow talk rubbish
21 year old colleg go girl live sion difficult scari affair almost everyday anytim danger place exactli outsid station underground bridg subway time light dark underground besid exit underground bridg walk toward shivar guest hous alway huge group men includ shop keeper sit along wholesal cloth shop go toward bmc quater make lewd remark also make sexual remark sometim even attempt unwant contact wors case wit action ever taken fix group alway gang entir area special night lane toward laxmi baug bmc quarter lot mini ladder gang sit soon sun set number increas time pass becom wors night
bangalor central cross street get bu 9 pm old drunk man stare pay attent busi street soon cross ran grab butt hand behind stood front distanc shock ye weak confront drunk scream shout help told men stand close bu conductor insid bu beat man misbehav came forward shame part entir incid nobodi came forward help
peopl bike tri come close friend pass harsh comment happen even
rel guy 20 year old 14 talk suddenli grope made sit lap know later told mother told wrong thing explain harass stuff
guy threw hot water say wear short cloth scare could believ happen
year ago around 1230 night walk bu stop toward home dingi lane walk look behind saw 23 drunk fellow whistl call name walk fast could board taxi reach home shatter devast want beat could alon
friend tri touch invit sex use harass go morn coach class even school premis
man start comment make face girl
6 peopl 2 bike 3 1 bike saw friend said someth realli bad shout comment us dont know place well wish go bata shop cinema dirti pictur area
realli bad
group boy comment start teas us way shop center gave attent catch way
young girl enter bu conductor make attempt touch waist comment quot god time make youquotquotlet go placequot
month ago return frm home saw group boy take pictur forc n girl sout
comment whistl
boy coment coleg girl 3 pm
got random messag facebook unknown person
catcal comment
even hour subject guy comment facial express stare catcal
boy comment behind group boy togeth comment
go market chain snatch two biker
friend stand two men start stare us make us uncomfort possibl us stand posit
citi joy turn horror spot even area like jadavour world famou jadavpur univers stall narrat event last year journalist fiance stalk assault bike gang near sulekha bazar around 1km jadavpur polic station tuesday night also earlier two drunken youth molest woman beat husband tri help garfa road citi polic seem struggl check grow menac eveteas molest area event add like chain snatch morn walker open fire near baghajatinreag park area though polic increas vigil yet event crop like rape teenag move cab kolkata add woe polit parti play earn vote bank fro
touch grope
went rajasthan trip went desert boy tri take pictur girl
went school enter class afraid driven class pay school fee sat behind school classroom cri told follow take home trick follow bush say short cut idea mind caress realis happen scream ran away
littl girl lull boy home use sweet succeed lull hous rape girl kill prepar bodi meat sold
travel dtc buse face harass almost everi peopl tri touch pass comment bodi stand
way home event male friend decid take gener compart instead ladi big mistak much crowd ok till get felt least three handsquotaccidentallyquot brush butt
go back home market two boy bike start follow
varieti demean behaviour face walk satya matreyi even
harass metro feeder
girl move road guy harass
go rel place afternoon man start follow hum song comment cloth start run went street lose
return coach auto driver stalk till home
poor light walk chan turn war catcal well
harass
walk guy pass comment
usual shop worker owner keep comment mostli incid happen night
happen colaba 2012 walk toward taj palac meet friend wear short spaghettistrap top cloth relev social condit make think singl man stop ask help phone frozen told take batteri walk taj approxim hour later cafe mondegar need use washroom reach washroom man exit male cubicl said hello rememb im phone guy gave pissoff getlost look proceed walk washroom held door said want talk push away walk use washroom could hear call friend say she came stand began ask add facebook realli angri told interest
niec sexual assault sent errand cloth torn harass rough handl tri escap comment hip figur would sleep
harass
go aunt home around 730pm guy start stalk pass lewd comment took auto left immedi
happen motil nehru colleg
go colleg old man dad age whistl
way purchas book bu wallet bag depart bu stand got know wallet bag
stalk leh market
boy group teas use bad word
harass morn even
age 15 yearstyp harassmentcatcal jammu 2016 timeeven
harass even
boy ratnapark make facial express push
time deepawali brother mine trust tri harass touch n lock room let go felt embarass shi n unsaf
whilst walk road almost dark alon suddenli guy came press breast shock could retali instantli
happen 19th feb went visit friend way back around 9 pm met group boy smoke wait taxi place quiet peopl gone bed walk start speak nose shut mouth carri uncomplet build remov cloth start rape turn turn
incid took place girl live near place backsid toilet complex 7 went boyfriend hous wherein rape boyfriend friend happen night around 6 month back
daili im get phone call number 8500330069 9705713510 8464014187 ask quothowmuch hourquot pleas someth stop nonsens
go colleg morn pass alon area kind jungl area saw man masturb saw start masturb activ
male studi grade 9 got comment whistl stalk
walk certain man tap buttock question start insult
pass two men remov privat part urin side road without care pass
got fight stranger catcal rude start use foul word
saw two boy attack two girl around corner butt
incid took place afternoon around 3pm metro red line inderlok comment upon
harass
man around 4050 year old comment pass store
walk station suddenli guy run hand back
guy look touch privat part sunday market masturb gross
rape teacher
accus snatch away victim jewelleri also made indec comment
go back hostel pg came across guy comment quot chinkyquot quotmaalquot could even stop stare complain polic incid seriou trust servic
man kept follow wednesday market buy veget market nuisanc occurr usual know
hi team yesterday survey tc powai near heritag garden found observ area cover tree area near garden amp mountain squar look like unsaf coz area garden look open wide clear see side
deliber touch claim crowd bu bu near arjun path
walk street group boy follow well comment vulgar word
group guy click pictur make video girl wear dress afternoon hour
yhan par kuch ladk khde rhte han ladkiyo ko dekh kar heran kart hae
ladi neightbour taken hous night 3 men rape behind hous threaten know whether report
famili member harrass multipl time
work cook bungalow near hous employ came kitchen touch breast told go away kept abus
morn travel bu guy squeez ass
rain heavili vehicl bu station person car stop stage driver told two girl board car taken karura forest rape
friend roam outsid main road night walk like could see rickshaw wallah saw us comment us
man first comment touch grope
 69% |##################################################                      |
travel two daughter bu mani peopl bu man came stood next elder daughter start rub arm got push
guy got metro kirti nagar sat besid suddenli start sing song look toward friend kept look reason talk us ask next metro station name etc call name get metro comment cloth also kept bump us intent
friend die hiv aid rape
shopkeep took girl photo saw confront
man first comment tri touch
go friend hous boy play foot ball ball came near came boy near get ball tri hug shout
area get photocopi around 4pm bunch guy kept whistl n make obscen gestur
man make weird kiss sound road
boy comment
follow us home market way make comment
way baba dogo clinic peopl sit along road hawk item stare peopl get even surpris leg walk quickli becom week laugh abl walk
wait metro 2 guy look start comment dress look
incid took place even two biker came bike snatch rs 17000 old ladi gun point
foot bridg trap sometim know wit indec exposur worst bit incid happen less ten second walk across foot bridg near kasturabai mrt station sunday even 7 30 pm decent look man call madam soon turn head toward voic expos privat part next second place walk casual like noth happen stood frozen ten second began compos alway scan foot bridg cross
incid took place 170914 around 24pm metro station friend usual get harass daili basi station would like safe citi take initi stop unsaf pleas
worst experi travel ponda panjim bu name jyoti ga 05 t6644 bu conductor came know half ticket passeng start say bad thing use bad word
wit incid guy comment 2 girl girl stop start laugh happen afternoon hour
hi im misbah incid happen 3 year ago left coach class vile parl travel bandra train 930pm stand parl station wait train platform wasnt crowd men stare pass comment top jean train arriv men gent compart hoot whistl sang bollywood item song harass disturb ladi stand next noth baffl experi rememb life
lack proper light area made unsaf walk metro station night
last year 17 may 2015 travel gorakhpur roadway bu mom appear examin mom sit 3 seat chair seat adjac vacant faizabad middl age man aboard bu sat next everyth fine realiz someth thought may due bu drive caus jerk reason happen sure man sit next touch breast inappropri sit fold hand may seen instantli ask leav sit somewher els react know happen made leav seat
train arriv 4 15amth tt way leer terribl gave stern look right old man come stair start comment catcal wink made face told fuck
boy usual get comment catcal facial express also follow man look wire scari
comment catcal whistl
elbow inappropri way
conductor bu comment dressup utter bad word
harass
travel bu girl forc harres boy bu
misbehav
group boy whistl girl walk alon toward home dark even
go market guy tri come realli close notic press peni waist
guy whistl randomli good look girl mostli afternoon hour
basic eve teas man tri follow
wait brother law unknown guy ask go
man seat next bu start talk ignor insult stupid prostitut dare beat iwa rescu passeng vehicl
guy tri rub bodi anoth woman metro station
group boy pass abus comment happen afternoon saket metro station
walk street comment quot big thing breast though childquot scold comment women near also scold
tri invit hous also tri touch
see guyz teas girl bharat talki squar
whistl
bike 3 boy came stop front friend grop breast went away
two boy made indec nois face
harass bu
friend told night went night club boyfriend around 1am danc went car guy forc romanc never want forc boy boy succeed sex tore cloth
cross road group guy take pictur
junior mine got marri complet 12th sad educ everyth today world marri girl complet educ mad
incid took place around mid april 2013 even guy wear black shirt black pant start touch girl tri cross road around bodi
chinki chikni kind comment
chain snatch comment ogl facial express other
whistl touch public bu
guy drunk tri touch realli offens behavior
travel drunk man total stuck ladi felt uncomfort unfortun ladi silent rais voic felt unabl help
catcal
teacher beat pupil
go rel home man tri take pictur
realli terribl
misbehav
stare inappropri way happen morn around 6 month back
group guy rie tocuh inappropri place whistl regularli
harass male teacher tuition alon student left
four guy comment girl
public bu saw boy ps tortur girl
donot fall defin criteria law appropri futur profession growth
pass church old delhi peopl start comment even hour
catcal
man touch thigh pack sit tempoo
shop sadar bazaar sister law muslim man appear accompani child slap rear bat hand hit man made excus say kid mistak lie face confront shout left right sister law stop pursu
boy tri touch deliber dtc bu
gone friend place afternoon sit outsid hous talk saw man constantli stare direct told uncl nearbi ignor us told aunti us came help us confront man first deni later apolog left
friend primari teacher school use behav ackwardli student would complain would beat without reason
walk street man came nearbi peopl road open pant show peni spoke dirti word run away
uncl want kiss promis buy anyth need
guy follow leav bag run away call friend pick
molest saw friend vehicl near babari mandi leav tri get hold pallu bike scold let go unfortun
forc touch train
sweep class room boy start run class touch breast
tow girl park lot crowd park guy contin follow girl irrit
way home school saw boy teas two girl girl wait bu stand bu girl seek help elderli person end
harass coach class delhi
group 5 6 men stand near cobbler stall observ constantli stare everi women amp girl top bottom access road
two boy comment friend
pupil come school boy girl walk togeth boy land girl breast ran away
return colleg rain heavili could find rickshaw decid walk societi 10 minut walk distanc metro station spot biker stop ask direct told start ask walk alon accompani ignor start walk revers bike said dont use umbrealla mayb boy get lucki see thing said bike could anyth also entir road empti moment thing want rush home
take pictur mundka
want cri pervert make comment
man stare continu
somebodi tri touch back
stalk white wagon r tint window
guy stand street threw water window went see happen saw masturb front us
guy park keep take round behind walk
chain snatch bhejanpura west ghonder delhi63 even
got harass rajiv chowk metro station
catcal comment night
chain snatach market peopl kept look know becam selfish peopl
23 men touch inappropri street
misbehaviour peopl
friend mine get back school stranger two wheeler stop ask direct place soon told smile point said quotdid see thisquot peni minut blank ran away
ladi hypnotis woke found haridwar without jewelleri
friend girl work cochin kerala get continuo call 918089417550 talk sexual abus word tri reach number pick call know call continuosli allow even use mobil purpos pleas help
guy came bike snactch phone
slip pretext save person part touch
case sexual harass done group young peopl metro train alleg victim rais voic
 70% |###################################################                     |
peopl follow way back home return job place market barnala punjab could report area surviv whole area report might gone away later creat problem us
come across fool vagabond almost everi place delhi colleg life pass indec comment whistl indec manner show vulgar gestur make person girl embarrass consciou
poor street light unsaf area near metro station
friend person start comment us
morn 2 men make nonsens drama front us take 100 rs us
guy pass filthi comment ask happen night
travel harass go anywher happen night around month back
young school children bulli outsid
survey carri safec red dot foundat along safec audit street market mumbai
comment ogl pictur taken night
aman tri touch appropri
300pm road quit silent boy bike snatch chain ran away
travellingin bu bu visit rel person frm bu tri talk n tri come close n tri touch shout n told got bu
girl spank butt walk could speak guy smile walk away
girl 14 rape
10th grade home class past 10 drop friend live lone area take turn lane least 5 dude 2 bike park soon saw start call name loudli say realli terribl stuff froze track got bike made way us start make nois scream loudli got scare scram yell parent come stare look boy dad drop bu stop
man call pass comment wait bu bu stop
stalk
earthquak peopl terai settl nayabasti peopl alway look differ know neg sing song look long time girl pass
heard number complaint femal student podar school today experienc mani men wander along santacruz w skywalk set eye girl begin follow extent ask think ignor walk
pass comment
got bu walk home auto driver companion follow 500 yard make indec comment thank back soon desert portion street end never get auto driver companion even insist short distanc
stranger tri touch friend intim area
teacher beat student
eve teas
return colleg use group boy tea shop smoke use teas comment sexi beauti well use whistl
comment girl afternoon buse
walk road man came girl touch comment sexi give chanc follow
teacher send student seduc sister
old man offer lift car work ogl
holi festiv tata sumo accid ran pedestrian cross hill road vehicl stop second quickli sped incid occur around 11am injur bit influenc time suffer broken ankl bystand rush nearbi hospit
shop mom went kurta store shopkeep ask measur first confirm size process tri abus touch privat organ got angri threw tape face scold
cross road zebra cross group guy alway pass bad comment comment dress realli disturb us time feel go colleg
wait colleg bu outsid home suddenli man came cycl touch breast went away could anyth happen sudden
use train travel work 3 month ago got train man made sexual obscen comment
man walk hiletol saw gang pass unnecessari comment girl pass rout
touch dtc buse bad intent
vulgar touch stupid bike rider expect illeg element caus troubl girl sometim peopl seen indescrib activ front girl hostel close proxim know fuck shit get area madhapur hyderabad ayappa societi around dena bank lane behind vodaphon store
whistl girl pass seelampur market night
touch
friend mine touch coach friend
harass
man near hous keep stare whenev step outsid pass comment place lewd song told want marri confid mom got realli angri start beat talk day girl alway blame even right hous bblock near wednesday market happen
go home friend boy sit stone start comment us alway soweto
return coach around 6 pm two boy kavi nagar bike pass comment firstli ignor start follow continu ask person detail refus tell took mobil phone pocket went 23 day continu stalk
touch grope afternoon
group boy comment sever women cross street outsid saket metro station
dec 16 case sunday got readi meet friend came flat fast food centr corner teh road pass food centr worker start sing song point cloth immdiat turn back taught good lesson shout told cant start sing song everi time girl pass street need control respect women guy stand head time made point
guy bu station comment like quotbabi babyquot even go far tryimg get hold
wait friend bu stand guy tri touch privat part happen three four time happen next
harass buse munirka night
take pictur without permiss
ladi want assist man instead took advantag situat request date
young boy look ladi sever year older weird way
go class group boy follow made almost cri
friend use face everyday class
drunk man kept tri touch girl inappropri bu
atm secur guard tri lock
go stage board bu dirti men kept stand besid move also move closer follow everywher move
incid sister know well start boy start follow first notic day past boy becom visibl enough stare amp follow till home didnt report anyon walk boy friend guy also saw slap even
men smoke bhang strate whistl uncomfort
kamla nagar market four boy start follow comment
rowdi look man snatch golden chain ran away could anyth
never harass experienc numer case market women harass custom market women harass foriegn white men come buy take snapshot without notic form mockeri cultur cameroonian
retutn pg around 8 even male follow comment even tri talk
ago travel kalyan station mall via riksha man around 40 sit next coerc travel tri touch inappropri tri confront put fight start abus
guy tri snatch money locket
harass
guy came car follow girl long comment join enjoy morn hour
kept laugh loudli attract attent made bad gestur relat woman bodi part possibl ignor seem mental
whistl
comment morn azadpur bu road
obtrus behaviour
eat restaur saw guy make vulgar express
girl stab man knife murga chowk street 8 afternoon
realli bad
group boy make nonsens facial express
guy follow us around khan market shop anoth park lot tell polic offic
travel public transport happen meet guy tri touch privat part
go railway station rickshaw bhaiya call friend say got someth asset u happen morn around 5
comment girl
crowd place one mostlu sexual assault happen crowd bu crowd street men take advantag women pinpoint assualt grope sexual languag often use
catcal whistl comment other
walk market serilingamp man walk purpos tri feel upon rais alarm man shout back use foul languag stander expect look
shop best friend random guy start eye us sing song
report big person teacher tell pass shortcut way get home faster
use go boy near hous use teas join name boy use stare comment n whistl
roam around guy start comment
want get physic
like go home stuff night like like late like around mayb like 1030ish late like went parti slosh like come home caus parent like strict ew rite take cab like cab driver start like comment cute dress like freak start yell shit like boyfriend tell like calm assclown even anyth im total gonna break anyway
friend mine child belong good famili parent work away home long hour left care trust male servant away would sexual harass hous
come back pg even saw random boy stand corner street start comment felt awkward ignor walk away
facial express
grope whistl comment masturb public peddar road bu stop
harass
friend boyfriend tri kiss refus accus steal
first guy comment suddenli snatch chain ran away
uncomfort touch auto thrice person harass twice avoid kandivali e station go thakur villag
two boy actual comment upon outfit girl embarrass happen even
report taken dadar street market exact locat bigg sale maitreen interviewe feel safe place due instanc stare touch
fell safe note face thing date today onward awar mani thing
 71% |###################################################                     |
comment pass near commun toilet group boy also stare
friend call men start touch
girl walk street boy touch ran away
chain snatch near kapashera border
man drunk place near place chef women go
walk street mind busi suddenli boy start whistl
incid happen friend go aunti place bu stop
ladi assault neighnour husband strip nake bthroom went ahead que
guy comment girl
stare us alway even invit us car
girl stay step mother father father brought brother 12year alway rape girl night sleep te boy sleep chair
uttam nagar termin bad boy comment onth peopl whomsoev goe special girl
guy comment girl
guy make catcal indec facial express
lost phone wallet
happen friend brother graduat immedi neighbor tri sexual abus escap inform parent big major incid prevent alert
walk home suddenli group boy touch person area
festiv season night time friend beaten gang boy
walk pratap nagar toward hous attend aakash coach class guy start follow made vulgar comment stop signal cross road notic tri take pictur thank polic offic near though noth shood away enough dog like
live foreign jodhpur rajasthan 3 month togeth face mani sexual harass situat summari walk even boy pass bike squeez breast moment pass happen friend well random boy street start follow walk ask hold hand hug stare comment everyday stuff mani time whistl happen also shout quoti want fuck youquot public alway went somewher peopl tri take photo men women ok ask ask permiss also mani time want photo ok also women want photo men
happen friend friend
wit situat visit friend mine behav strang funni tri kiss took cloth emabarass
classmat look funni way whenev eye meet wink start make funni facial express
boy pull even time
way tuition harass
famili friend tri inappropri touchi famili gather stop report parent cut tie
way back home work aound 7 30 pm stranger whistl tri ignor came straight told quotyou comingquot scare
way school saw group boy smoke bhang got afraid saw start chase insult marri
harass coach class delhi even delhi high court
afternoon happen
hang 2 friend karol bagh market amp 34 guy follow us pass nasti comment
go even boy start troubl even call name start comment
peopl metro tri push touch happen afternoon even
ogl
bu feel unsaf stranger push bu touch stand bu well guy want touch bodi part
harass
ladi dc stage around 500 travel rural home men caught shout help came rescu peopl woke found ladi rape men ran away luggag
chain snatch rickshaw
misbehav
happen train cst panvel 7th 8th std gener compart train guy inappropri move finger stand near
travel public transport 2 boy sit front constantli stare whistl nobodi around anyth
start notic man jasola metro station ogl coupl day knew station would usual get day made extra effort follow rout complain metro offici
sexual gestur comment whistl
happen last month friend man came near tri physic
walk undugu school st john primari school group af men ask could escort said start follow reach home stop follow
stare irrelev unbear unnecessari comment
age 18 year oldtyp harassmentstalk ibex coloni leh time catcal skalzangl taxi stand leh time comment deymo thong duk look nice time sexual invit ye time
comment hoot inappropri action
incid 19 year ago friend sexual expliot uncl
go school stranger com front hit
guy take pic
birth went friend way back night men caught rape
sing songsvulgar commentswhistl press bodi public transportand touch
catcal comment group daili occur
peopl click pictur without permiss
nahargargh fort jaipur guy wink smile reason made feel awkward
stalk
afternoon 3 30 pm
come back home school friend afternoon men phone click pictur friend happen near jb 454 front black school build
scooti wait signal turn green group 6 guy start pass indec comment
incid took place janpath afternoon around 2pm shop road side seller sell belt forc us buy belt refus start comment bad way person like bad experi
guy make indec comment whistl
travel bu even hour drunk guy tri touch continu even repeat attempt stop
guy resid near charni road night incid wait friend suddenli person came start stalk mr went later came 2 peopl start stalk
touch school boy public transport
survey carri safec red dot foundat along safec audit street market mumbai
indec behaviour well read boy even
walk road guy build balconi whistl call hous area student differ place live hostel prepar competit exam wear indec cloth infact wear salwar
biker comment whistl
happen saw boy time tihar ogl well clap girl roof van came play deusi group sing song also saw girl scare n seriou condit poor
way work cross road biker collid thrown road group come help thought help start touch grope odd place tri push crowd someon conveni taken phone manag pull crowd emot hurt physic
pass boy start say like tight dress
street get harass daili guy comment boob buttock say sex touch indec alway run street light
show adult film class mate
kidnap rape
harass publicli lokhandwala afternoon quit part road noon said word tri confront two boy bike rode whistl
holi someon threw egg near lower back realli embarrass
travel delhi metro man stare girl intent full lust decent amount time could take stood point view gave casual look fortun sent messag
boy tri touch bu
group random stranger stare friend bad way
girl campu want make love without accept request
go tuition group boy whistl ogl inappropri way
guy follow near park sanjay camp
boy lie took friend slept 3 day mum complain threw hous night
man sodomis buttock night
everywher locat subject comment ogl etc
follow car
travel metro friend three boy stare start comment disturb friend also start follow us ran
travel bu someon stole purs colleg id librari card import document
guy stalk comment go home colleg
saw women wear short dress suddenli old guy tri touch tight
harass
larg group men good intent
walk sister man click photo sent photo us facebook say bad word threaten us
touch share autotouch bandra bu station local train
go home colleg boy toch shoulder tri come close
walk street
ogl travel train person repeat daili 811 local train thane cst
go work mine goon type peopl sit vehicl pass start follow pass comment final isol tri grope shout made nois make run away
stalk guy teas
catcal comment indec exposur
friend eat icecream group boy pass us went littl far start stare us minut left place
 74% |#####################################################                   |
guy comment cheap
usual get whistl unnecessari facial express travel public vehicl colleg home vise versa
say go school noth instead wast parent money
friday sport school friend go home met boy gogo call friend ignor start abus
man walk street laini saba wear silver chain familiar are boy came statch chain neck ran away
poor light condit area give rise mani petti crime girl societi
hi follow incid happen twice walk back home walk lane adjac kasarvadavli naka lead toward thane muncip garden side 8 10 pm boy pass toward direct walk stop check right face drove ahead gave firm rude look get verbal fight lane lit peopl pass surpris thing dress salwar kameez occas alert cheap liqour shack prior hawar residenti project via lane walk boy head direct mean either belong slum nearbi head chill liquor shack alert two place ahead liqour shack desert notic rickshaw driver local boy frequent place liqour consum open seen road get spilt two direct hawar proect head valley like place desert polic need presenc felt alert three lane half lit spot complet unlit street light pleas someth asap rape crime take place nearest polic station kasarvadavali polic station mani thank websit
five year old girl defil neighbour trick join cup tea hous defil babi told tell anyon
realli bad alway happen almost crowd
happen 9th class privat tutor teach math ask watch internet start touch thigh felt uncomfort ran went home told parent scold told never share incid
teacher beat student
certain ladi walk roadsid man bike stare said sexi look good bed
come back kamla nagar market 700 pm guy tri follow kind eve teas
pedicab cyclist follow behind pedestrian block way grab expect quietli give
pool guy tri touch unnecessarili even hour
come danc class guy bike follow till home comment
incid took place metro afternoon someon take pictur girl metro
parti villag night stare
two men chain snatch necklac old ladi guard street light present
boy comment sleep terrac construct multiplex
group drug addict gambler alcohol hang around road lead societi lower parel station make unsaf take rout especi dark
took pictur without permiss
man wa follow shop came touch butt tri ran away
walk road go class
went bengali market around 9 purchas medicin viral need antibiot way back heard somebodi whisper around ear look around found boy roughli age group 1214 ride bicycl went ahead kept look back make gestur second later return repeat make third attempt pick phone make call see hurri mandi hous golambar ignor first time whisper comment dare second time almost infuri alon huge lane around ficci complet dark desert except ice cream vendor men loiter around could easili thrash boy stop fact know ugli turn situat would take state shock could believ boy 12 yr would dare someth like ironi law student advoc right juvenil paper
wit flasher car guy pass lewd comment wait metro station
ogl facial express touch grope other
friend mine child belong good famili parent work away hour left hand male servant away would sexual harass home
man drunk felt ran happen even
stand metro someon touch hand smirk told deliber
incid took place near kamla nagar market front kirori mal colleg even return place middl age man sit front kirori mal colleg suddenli start comment along obscen facial express sexual invit tri ignor first kept person point time panick ran away start come near
happen year back go center guy would follow scare alway ignor receiv call idea receiv number would send messag insist muster courag call told stop troubl even abus threaten inform polic
wrongli behav
girl touch boy public transport feel soo uncomfort
stalk
public vehicl boy touch
group boy pass comment happen even
incid took place manav chowk red light stand opposit red light free left turn took place around 57 minut ago mean around 93540am wait auto amd polic innova take left person man sit insid wink person polic uniform drive polic innova guess polic offici shock peopl given respons protect tge societi someth wrong system highli highli disappoint angri shock
began come back school decid stop friend place check home decid open door sinc key minut later male friend came look saw boy start tell beauti realis get unbear escap
sister catcal market
walk man follow start call
6 year father friend usual came home came alon tri rape shout friend came ran away
catcal boy coloni
indec song
sister get back tuition center hit man back side ran away teari told
pass comment fight
guy comment upon girl travel gener compart afternoon
bad crowd stare make indec bodi movement
incid took place near stretch highway colleg vip pitampura pitmapura metro station around 13th 20th septemb happen afternoon person pass lewd comment anoth blew kiss
park play basketbal friend notic take pictur friend
incid took place gtb metro station afternoon boy follow distanc say hello ignor
happen night janmashtmi went templ man touch chest year
went watch movi neighbour surpris movi show nake peopl sex saw ran away
guy knew grope ass 11 year old
even tri touch
friend good time suddenli us realis group boy behind us take pictur us friend walk ask flat refus confess like even discreet ipad open point straight us demand see album found mani pictur video us scari made delet
light miss peopl follow us
learn ride 2 wheeler 3 boy use comment daili
conductor like touch girl privat part give bck chang
group guy teas everi girl pass way
2 men black pulsar bike snatch woman chain zoom away tri call help vain
comment catcal whistl touch
girl ran away home go sleep men
marri man area continu stare smile wrong intent tri ignor follow said parent solv problem
incid took place nearramlila maidan area landmarkgarima vihar ntpc township walk road front hous auto driver pee side saw start shout bad vulgar word
wear short thane station stand near polic desk awar someth might happen man pass tri pinch thigh
went shop famili colaba causeway shop 23 pass lewd comment sing
way go shop metroo station crowd enter gener compart somebodi touch behind
person bu stare continuosli afraid
guy comment usual go colleg
man intent touch chest insid bu bu pack could anyth man look fear
stalk man look wire scari share friend famili
teas comment girl way home
stare pass comment
two comment stare upon
girl stop man proceed seduc scare know
group guy lurk around theater night call girl
sheepish look guy felt friend hand genit crowd metro noida citi centr metro station
whenev go terrac spot guy stare continu even go friend like hous sainik enclav happen
guy sit bu stand opposit mh saw start whistl speak someth fast smark lip pout wink
indec exposur
aunt go road peopl take pictur us aunt decid go away happen month august even
man came start harrass go lone road
street light
man touch back
friend go anoth friend home
school boy follow return home school
walk nih colleg guy whistl
sister call unknown men rape kill sad ordeal
show bad facial express walk matur guy unknown person never seen
sent daughter charg phone neighbour place rape feld sad gave money convinc speak guilti charg
go home 9 pm mani girl school side alway kid feel difficult feel uncomfort
friend talk came boy comment us sexi even tri touch us old ladi scold boy threaten us tell us wait watch
biker tri grab breast
 75% |######################################################                  |
guy caught hold hand walk even also encount lot comment catcal
due lack street light difficult go alon 8 mani passerbi take advantag
harass
incid took place rajiv chowk metro station even pass station toward platform noida citi centr saw behind toilet area comment catcal took place
afternoon 1 30 pm
group misbehav
friend go school earli morn bu man sit besid friend show privat organ felt awkward difficult got bu
follow group boy
chain snatch
man work factori near workplac use follow auto regularli
go fetch water near home white man came pull touch breast manag escap
boy caught unknown men sodomis threaten knife
come mother palc work met boy influenc drug greet will respond made start cali prostitut
yesterday dat 9 03 2013 scooti travel granni home way two boy van start follow tri teas drive vehicl parralel scooti scare till 10 km follow
whistl teas visit place
boy bu pass bad comment group girl say thing figur cloth
men follow forc give number gave mother number
hous owner pass indec comment happen even
chain snatch incid wit morn
poor street light result guy tri grop
boy go bike pass cheap comment stare
2 men start pass comment pass
take pictur comment girl come back offic
three boy bike start harrass cab way colleg note number plate gave polic
mth road start ambattur industri estat bu stand area avoid place especi even time seen heard mani incid misbehav local men girl women suffer due lack street light make incid happen conveni
day ago go home boy even age peopl start comment friend
stalk stare touch invit
go tot foot group guy bike harass us
jaipur colleg backpack work went shop pump 45 year old sikh man splender bike came directli ask sex hi 2 variou app like webchat showoff littl bi approach ask gay relat unless strong punish live non bailabl prison min 2025 year death punish rape happen happen still happen awar come punish practic
inappropri comment veget market around 4 pm
tri pull jacket near fire station roop nagar
follow ogl common panvel
slap boy
touch even bu
sexual invit
return home alon met two boy wistl comment
park felt guy stalk feel safe
sister teas group boy
bu ladi stand ladi bend pick someth time man took pictur pretend take selfi
stare comment
sister go home saw girl wear short skirt teas group boy well saw boy tri touch hand girl fought back slap boy
comment
cross road buspark area man came touch chest tightli happen sudden
somebodi stole stole wallet phone night time dtc bu 729
happen girl child age 6 sexual abus 3o year old quarter villag grew
bu man teas whistl
guy around 4 pm stand bu stop cross road start click photo went slap
travel home bu realli pack guy next touch tri grap hand
bakeri man enter bakeri walk straight girl told much love girl walk didnt know guy
get home tuition notic follow group men confront deni follow happen afternoon
tailor tri get close take measur fit
touch grope ghatkopar even
park scooti soon remov helmet man pass comment move forward high speed
ogl facial express
boy call girl told beauti would want bed
even guy alway stalk way colleg annoy
sexual invit 3rd person
harass daili night
bu 764 near iit
man came close friend bu touch privat part suppos touch bu stop near connaught place
area stay pathet alway boy group stand road side play card pass comment passer tri ignor comment time feel ignor like girl like easi target road side romeo talk parent ask ignor somehow like ignor want take strict action boy realli pass dirti comment girl therefor parent allow us move hous 7 pm
two guy scooti pass front cross lane whistl said quotmeowwquot
happen someon know travel metro crowd spank
walk guy 2 guy comment us
touch inappropri way
incid took place 29th juli 2013 near m2k rohini even two person bike snatch chain ladi ask address run made bad comment
invit male friend date cours drink drug rape friend also three friend frind gang rape
even follow guy whistl tri grab attent way
man show peni call thought call reason turn back show peni
report experi friend walk bu stand everi morn catch morn bu offic way often come across middl age men ogl women time occasion budg elbow tri touch pleas avoid walk pavement
afternoon travel gener coach subject indec touch much peopl even asham
40 year old man stare bu felt uncomfort
happen metro feeder naraina vihar even
shock incid report pink citi eve teas common happen girl almost everi well let share 1 incid friend studi 1 prestigi girl colleg jaipur stay pay guest cscheme jaipur two friend went market get someth suddenli 1 car came think 45 boy car tri drag friend car grope wrong place luckili friend save shout dread still cannt forget incid 1 case kind incid keep happen
incid took place morn bu peopl touch privat part girl take advantag crowd bu
first class ladi compart alon man compart start sing song doubl mean
age 15 yearstyp harassmentstar leh market februari 2015 time stalk leh market februari 2015 time catcal leh 2016 time even other give phone number pleas time
man high drug alcohol came ahead touch inappropri amidst crowd kurla station stop beaten anoth guy still audac turn around wink left
realli bad
indian men total creep everyon stare much feel violat walk mumbai monday outsid domest airport congress goon rep bike made want hit
biker took round road approach touch happen afternoon
walk way gumtai chowk boy pass comment quotyou look beautifulquot well whistl
friend face
hapen afternoon
friend face book send text unnecessarili
saw man tri rub privat girl back
teacher beat pupil
privat tutor touch pubic area start rub vigor ask mean romanc class six student scare tell parent
 76% |######################################################                  |
happen road even wait auto time group boy pass nuisanc type comment
friend got invit femal colleagu sex
stalk day
come home work man street ask sit car go
man first comment pass touch inappropri
3 guy tri sexual abus friend
wit chain snatch abus situat wherein woman push away abus also chain snatch ran away could stop
walk home take groceri patch road connect pramukh vihar societi main road time 0740 pm 2 person bike pass make comment time came back pass comment tri touch shout ran away
night guy call friend call home tri rape ran away mental trauma
boy start masturb front us
sent father pick sister school man stop amp tie kiss ran away
boy sell mandazi morn call whistl girl go school
boy whisl go home school felt bad could react
man shout friend appear go market
follow 5 boy kept laugh pass comment
follow
men drink alcohol woman came man begun open button touch breast
guy bike
incid took place krishna nagar market afternoon friend forc car gangrap
ladi go back offic place even heard somebodi whistl pay much attent late want reach hous fast earli
friend badli haress four boy return home colleg
stranger click pictur know constantli focus camera toward
even group boy take pictur girl
children 1213 year old use abus languag return colleg hostel friend
person comment mai amit mujh pehchana kya amp assault girl
night group boy stalk start run safeti drag start shout crowd gather ran away
walk road boy stare comment
follow boy call ignor ran
girl go metro station saw somebodi stalk talk somebodi phone follow 20 minut met friend go boy disappear
boy car comment friend jog tri follow us
grope touch dwarka metro station
man stand besid bu nepal yatayat tri touch hand felt uncomfort
incid took place outsid vishal cinema around 7pm observ group peopl stand includ girl smoke pass remark girl stare badli
car stop friend jog start call sexual invit
stare near jnv hostel near saboo villag ladakh
night invad hous thought alon luckili dad around came room tri touch walk start scream dad came late ran dad reach
9 year old sexual harass salesman could say anyth build search could find
friend buy veget unknown peopl stalk us way pass comment
wait riskshaw rithala metro station comment upon call slut
take pictur wrong place
boy bike pass filthi comment tri click pictur happen even
make bad facial express toward girl stand dtc bu 543
walk road saw girl pass dress short dress boy call told go greet boy refus leav hand
comment bad word
go toilet complex notic guy stare look start make indec gestur use toilet went market mother notic guy went away saw mother around
cheap comment pass guy
catcal whistl comment
iv trip colleg rajasthan head depart touch inappropri
man daili ask lift start stalk
catcal whistl
took pictur tri touch randomli happen crowd train buse eve teas anoth matter look
went male friend place visit went drink spot return place still gave someth drink got drunk made advanc toward respond drunk
younger brother walk street boy follow us scare whistl start walk briskli walk fastli crowd area stop follow us
bu man sit next tri touch bodi feel nervou uncomfort
mom come back friend place two guy bike start circl us dark gave opportun scare us
friend harass public
men comment appear
saku return back work aa group four guy bike pass comment whistl disrespect express
incid took place front banasthali public school morn woman walk biker snatch chain purs
friend mother singl parent neighbour stalk comment rough word use alon
comment catcal whistl
return colleg bu stop guy stare continu scare
man masturb near station leer
go alon uncl home group boy start use rough word pass
comment
student regular travel never felt safe face assault almost everyday
realli bad
famou phadk road dombivali east proud peopl dombivali everi even especi sunday market harass victim group boy girl come mani reason boy stand group road teas girl nobodi complain
guy make inappropri sound face happen even
comment catcal
person travel bike 300 stop rob wallet chain
everyday gang boy sit exactli outsid entranc chawl 7 1130 pm keep harass girl make dirti comment make life miser scare pleas helpexactli outsid gate bmc chawl laxmi baug sion near shivar guest hous opp sion station
indec exposur also face even 29 sept 2013
go bu ratnapark jwalakhel touch guy thigh move anoth sit
ogl pictur taken
go shop saw 2 boy sing whistl girl start walk faster avoid
touch grope stalk catcal whistl delhi
navratri pandal 2010
harass offic
walk way friend home saw boy take pictur girl without consent
random guy start take pictur
man follow
2 guy drunk grab shout help guy came rescu
steal
harass street boy harass school teacher harass home peopl live bring report card hous mother told teacher went collect report card beg teacher refus still ask class master even refus let attend class sinc attend class way studi
wait bu boy stare quit long time made uncomfort
chase group boy scari final manag came
harass
sexual harass ub femal friend friend mine alway expos pubic hair breast time time see alway wrong notion sex maniac
unsaf sub way poorli lit light without secur monitor danfer night time
husband came hous drunk ask give money welfar children start beat tell think bank pack live
bu
hous next idbi bank care taker seen lean entranc gate hous keep feel p see women pass women shi away cant stand disgust guy sinc 2006
group boy age group 2030yr stand amp pase comment group friend near citi resid durgapur w bengal
travel old guy nearli father age sat next bu start start talk phone bag lap suddenli thigh sens someon touch thought bag reach destin amp took bag lap hand thigh
teas way buy thing
inappropri light road block multipl riskshawwalah loiter around
come back colleg often group boy stare comment girl friend face problem approxim daili basi
boy like whistl us way schol
group men whistl pass comment park near school school near kamand road
shamelssli click pictur women
 78% |########################################################                |
gotten bu kibera tout tell size tell bedroom size
four year ago form three friend mine come back school attend preep even school read 5 pm reach central market former c c banana plantat two men came plantat held drag bush rape abandon later discov three day
shangrila resort grope men pool hit wave lost balanc drown
harass afternoon hauz kha villag
follow boy
guy use stalk kept pass comment whenev saw stay satyaniketan si could avoid
comment catcal whistl touch
near tuition center shop would buy stuff class guy would alway time would pass comment cloth
head bandra wait train man blcak shirt stare right us quit time shamelessli click pictur us
even chain phone snatch
cat call jeer
come church friend reach dc ground certain man strted call us refus hurl insult us
happen monday even unfat wit sexual harass boy girl boy girl stand street hous suddenli pass front saw boy touch girl bad way girl shout boy leav alon greatest supris turn behind saw girl alreadi nake floor guy nake
jaipur famili holiday shop local market jaipur felt somebodi squeez butt broad daylight crowd market place wear full skirt proper tshirt even quotindecentlyquot dress turn around slap guy regret frenzi occur report
guy teas whistl whenev go shop
hoemtown hyderabad walk cousin somebodi came ask address start show nude pic men women class 7 got scare ran away
way scooti supermarket two boy snatch chain
boy pass lewd comment pass happen afternoon hour
bike rider circl around girl pass variou comment happen even
even metro station guy slap girl suddenli without reason
complet slum area fill drunkard drug addict amp area lot murder report day bewar go next sea amp could thrown sea tortur help
guy stand bush said someth look peni
harrow experi
eve teas
touch
happen metro
uncl scooter follow suddenli stop front pass lewd comment disgust expos privat part right
small girl beaten man thick stick
girl touch got stuck crowd men
person take pictur privat part girl walk metro station
guy girl comment
pass katwekera man call told love good shape
man make inappropri action
recent rickshaw driver start shout comment think drunk scold sister ea disturb
unnecessari comment group colleg boy everi morn make feel insecur
person stranger follow girl skywalk
old man promis buy touch screen phone accept sleep
comu schol head home met group boy call us go gret refus start follow us home
comment catcal sexual invit much ogl stalk shop mother earli afternoon group men gather block way stare laugh make comment intimid tourist group gather 4pm especi bad near store cafe near park lot wait cab tri go cafe avoid men felt unsaf shop got difficult find way around get direct help difficult alley fill men
3 men comment friend
comment amp sexual invit munirka
stare shopkeeep kept look breast
stalk 2 men
2 student go colleg ride public vehicl feel guy tri touch bodi part
colleg premis ask sex afternoon
wink fli kiss
incid took place train get train man offer carri luggag purpos rais hand touch took time know what happen
rape commun indira nagar
incid took place near srcc road place unsaf night mani terribl thing happen
walk kalyan station guy tri touch stomach
jubile hill road number 10 extrem dark road walk absolut alon must carri pepperspray hand walk even catch autorickshaw biker catcal whistl relentlessli
group peopl stare wrong way also click pictur
old man came touch butt walkin mum sister
go shop night certai man show money told tha want girlfriend
mother go market chain snatcher came stole chain neck
travel public vehicl got blank call messeg
two guy follow friend time pass comment
metro station crowd grope
ask two secur guard rout easi coach bu terminul ask phone number first get rout give ask date begun follow went stop told marri believ kept walk
boy stand outsid school hastsal keep whistl comment girl come school even teacher say anyth tell us ignor
school girl pass carri luggag school nissan 2 tout start enjoy fat ask go type bodi rather get marri earlier girl ignor
boy touch preciou part girl intent ramja fest
boy teas gurl seem helpless even could much grp guy pass quietli
shop mall friend went washroom suddenli saw fellow look window call polic howev fellow seen anywher
peopl like talk behind friend beacaus like tight dress
10th januari around 330pm delhi metro person stare wrongli start comment
sat next man bu put hand thigh touch felt bad decid chang seat
mother sent night boy greet refus answer start touch breast
walk alon guy came ask address phone number even forc give number
boy misbehav horribl experi
scari experi theater premis today wait friend sahakar theater chembur man stand behind back pinch bottom ran away left miser feel
travel bu man continu stare stare weird told friend next scold
boy make foul comment catcal happen afternoon
walk home around 1130 night male friend motorcyclist quickli drove past grab breast knew happen unabl get plate number descript wear helmet
come colleg guy stop car start comment like quot umm hmm like tea come sit etcquot
incid took place govindpuri kalkaji lane 5 market afternoon wherein notic boy harass girl took hand start sing song also use inappropri word boy question boy start beat month
friend went meet friend friend good friend took hotel beer rape
stand alon sudden guy bike start comment ignor start roam comment
touch possibl person area bodi guy rub hand butt time tri look insid top happen subway near cst station dark
guy make group teas everi girl pass way
guy follow mani day told mother told father policeman father guy never follow co dad ban
930 pm earli morn near kanjumarg railway station gang stabber rob harass women girl
wit incid sexual invit afternoon hour victim bike
tri cross road guy bike snatch girl chain ran away happen afternoon
3 guy comment tri take pictur metro station
boy comment whistl
go home wheni saw boy car harass girl
travel metro crowd someon tri grope genit
run stall basic good till 10 pm everi night night around 10 pm pack go home man came start make sexual invit went away threaten call polic
two guy take pictur comment bad word public bu
week ago return night parti got rickshaw right outsid build drunkard randomli stood next stare continu rick guy ask rick guy wait till saw enter build wing leav ran life
man touch festiv indrajatra man polic
boy tri persuad girl sex sexual invit want go home
lone road
guy make face
 79% |########################################################                |
wear short walk near armi area raini road desert 3 guy bike stalk comment even tri assault alon helpless
whenev girl pass especi even hour guy alway comment whistl
walk around market shop mg sister random stranger pass spoke ear say quotkya matak rahi haiquot
group boy use abus word
get back home uttam nagar east metro station follow two men bike street light around
rush tri catch train near virar west railway station boy came front bang hurri catch train tri take advantag situat luck good stop give tight slap arrog twist palm shout take polic station time realiz becom panic want wast time left behind walk way
four five recent event violenc abus public reaction merit comment first amaz distinct drawn mani languag use insid parliament outsid let begin unreservedli condemn defenc minist khwaja asif comment opposit leader shireen mazari voic rhetor request make voic feminin use term truck trolley moham hanif insight piec subject correctli identifi remark hark patriarch expect everyth woman conform standard set male men societi feel dictat term behaviour dress even tenor voic woman anyon els would bad enough member parliament cabinet minist everyon expect higher standard restraint control anger howev serious debat protest parliament make littl perplex parliamentarian indulg see much wors languag behaviour outsid parliament public space nation televis social media expect creep hous arent peopl talk outsid insid hous inde arent support part societi dismay surpris alterc hous ms mazari khwaja asif surpris indign given indign peopl know well low standard kind discours plung around us pti arm right rightli never bother condemn indign utterli obscen abus hurl sister fellow citizen oppos polit wit languag abus misogyni social media televis pti leader languag abus male oppon atop contain televis show secret either henc feel need repeat parti guilti similar behaviour though cesspit extent certainli guilti act surpris someth like happen floor hous condemn boycot play activ part drag standard discours low afraid even speak fear abus guilti parti certainli media everyon know channel show host love creat situat might give dividend shape abus salaci slur slap punch glass water coffe thrown claim boycott channel show everi mna mpa advisor activist economist etc still behav like media whore grace show host know encourag abus fight bad languag obtain rate act sanctimoni nowit appal lead light sever opposit parti continu spend energi time use media energi time call tor panama leak commiss seem noth els happen countri everyday report woman burnt aliv honour equal horrifi seriou issu could merit attent self serv focu go prime minist exclus duti disgust say least one ear ooz pu nonstop month long hyena shriek press confer appear tv show panama tor fourth girl four day murder leader senat raza rabbani took notic task senat someth everyon made caus level apathi toward plight peopl gener grow violenc toward women ruler apathi result impun violenc women exactli reason senat hamdullah cant believ senat jui thought could presenc sever peopl studio nation televis level explicit pornograph filthi abus well known right defend columnist marvi sirm also thought could beat tri public demonstr everyon level apathi toward violenc women countri demonstr everyon go behind close door man thought could would get away tell someth state societi fearless jui sure support rule parti govern befor assault marvi sirm hafiz uzaifa shakir juif yesterday releas video say hum pashtun gali ka jawab goli se dete hein pashtun respond abus bullet tragic heartbreak singl pmln leader publicli condemn jui senat reason polit expedi brave women like marvi sirm stand support elect govern peopl like stand protect democrat project democrat support protect forsak us crunch time true test democrat norm valu
guy whistl
friend colleg walk alon tuition class afternoon bike came opposit side touch inappropri sped away citi centr durgapur
bu panjim ponda man start follow came bu bu crowd sometim got seat came stood next start touch privat part open chain pant start use indec word
guy near 69 bu stop follow way home
friend comment group boy ignor walk past
four fren walk kanti road lazimpat saw basketbal match minut walk nearbi garden kanti hospit garden stranger boy play footbal invit us join deni walk away follow us later start ask money us ignor situat becam worst start touch us end big fight
school girl corner roug boy way home lunch boy touch breast comment quotthes one need taken care ofquot girl say lucki
stalk
catcal gk
uncl harass brother daughter child girl saw neighbour
ladi call men public felt asham
stalk secual invit
gone fetch water man approach ask water give scan top toe stare made extrem uncomfort happen near wednesday market
return colleg old guy whistl near poolchowk facial express unus
crowd peak hour
guy touch back even
incid took place afternoon around 215pm jan 2013 went friend visit lotu templ saw random guy snatch ladi chain
colleagu ask go kathmandu valley reject stop talk
group boy seem noncommut stand daili near ticket counter pass sexual colour remark girl women pass way tri report polic seem disappear day show back presenc
mangulsutra snatch 2 peopl ride motorcycl drive wrong direct initi went littl away victim enough speed time ride correct direct pass sit behind struck chest look like bang chanc later realiz mangulsutra snatch leav behind small strand hand polic station 300m rang victim file complaint immedi polic attempt catch thiev blow woman chest hard might cost health even life
realli bad
poor street light near metro station unsaf area girl even broad daylight
due differ look boy pass bad comment around 5 pm near indravihar
basic grope blame us
survey carri safec red dot foundat along safec audit street market mumbai
rememb group boy follow bu stop resid shit scare kept pass lewd remark helpless
passerbi intent bump tri touch grope wrong place
girl took lift brother got harass bicycl 12 year old girl return place around 6 oclock even brother offer lift front sit minut touch massag breast say feel comfort
continu comment came way earli morn hour
man call invit home thought classmat saw realis stranger
station light poor also lack proper sign lead emerg exit
man tri give signal inappropri manner
even two men tri touch girl much
harass rajiv chowk metro station
male tout tri convinc femal custom enter bu refus start call name say stupid like even husband tire
street light unsaf go dark
got pack bu take advantag situat start push elbow toward intent
go back home 2 boy start whistl
chain snatch near bu stop came help
hous outsid wash dish guy make unpleas sound call attent final walk said want quotfuckquot like could anyth peopl home
crowdi area touch incid common
mapusa pernem buse
saw anoth man touch woman buttock
happen p g p garden near market
two guy go scooti slap touch back move vehicl full speed
catcal comment touch ogl dtc bu bu stop dhaula kuan hour night
need go mri scan rel upsana hospit kerala need chang cloth person open door mri scan ill person door close forget lock person call travel hospit person auto rickshaw open purpos old men stand next fill crowd
girl waa period bioy forc sex
feel unsaf travel alon incid
lot facial express like fli kiss came way travel metro even hour
girl follow silent road comment say know ask ph number
even shop shopkeep would stop comment
happen night
use drop brother tuition center blocka guy might follow saw sever time stare walk along approach ask friend say like lot told troubl kept persist scare chang brother center end start appear street live
comment look bodi part
two guy came bike snatch chain
boy take pictur loo
girlfriend comment upon return colleg happen 11th septemb 2013 even around 530
come back hostel two boy bike pass cheap comment could respond bike pass away
incid took place around afternoon wrap face cloth start sing stupid song ask show face
realli dark 2 school boy tri grope start scream ran away know school boy carri school bag
happen club someon tri take pictur
incid took place even group boy car ask loud come sit car even open car door incid take place dark place street even openli crowd place
poor light area therefor night becom difficult girl go hous
harass
guy indec expos privat part even know girl around would notic happen outsid wadia colleg bu stop
happen night friend come back drink spot boy persuad sit discuss go home girl refus seat discuss boy start harass
misbehav
visit guy shut door succeed escap
poor street light
alway beliv ask direct person peopl like secur offic ask traffic polic offic direct assist ask phone number keep invit date man look like 50 year probabl children age even older would imagin mistress man old enough father
bu friend person start comment us
place near nandan inspera datta mandir often dimli lit usual group hang around area eveteas group activ noth els help polic patrol seen anywher encourag guy polit connect builder acquait sinc consequ peopl alway get away night patrol kind surpris check done random feedback ladi stay near area taken even templ safe visit day
boy teas girl girl colleg compound
board privat bu tution center even conductor refus take money even protest stop came got conductor pass lewd comment told brother incid drop pick tution center
afternoon time brother sister guy pass us boy touch butt inform brother brother inform polic polic made sorri
age 18 year oldtyp harassmentstalk ibex coloni leh time catcal skalzangl taxi stand leh time comment deymo thong duk look nice time sexual invit ye time
way home school boy teas n pass comment
person kept cat call pass lewd remark friend confront resort violenc
 80% |#########################################################               |
touch andheri station
touch inappropri
afternoon friend group guy continu comment upon us tri touch us
live hous guy call room canopi show someth got close door start plead sex refus insist 30 minut seriou refus later open door
friend went outing boyfriend valentin group guy teas friend boy friend went oif place without word
young girl 9th standard play street residenti area bunch friend man possibl late 30 earli 40 ride scooter came close moment realiz zipper trouser open possibl stroke full view group 34 teenag girl shock hurri hous minut silenc realiz occur vow take registr number vehicl next time happen never got chanc
boy whistl friend go home
men road stare
boy school dress pass comment
two person came bike threw water balloon wet complet
lil si alsway harassr school frm teacher tell mother n complain school n school restig
wait station guy behind start get closer kept brush hand waist arm first tri move away continu immedi gave look told straight away
visit place bhaktapur bu suddenli boy enter bu besid simpl type girl sit boy came near girl an speak tell vulgar word also feel shi
walk home even man didnt grab hand want touch privat part fought fell ditch felt bad dirti smelli
go hostel group guy follow teas felt sad
morn two guy contion journey tri touch commentibg make face un comfort int
touch inappropri place push deliber
bu walk street
happen last month go back home colleg rain heavili train empti enter train occur person dabba anoth boggi besid men stand cours travel could sens someon stare continu look around saw guy around late 20 stare even look wont stop stare would give ugli smile would freak eventu start ignor
girl stalk tri molest
two big men around age 29 grab hand girl lock rest room railway station secur effici help
market girl buy undergar person work ant 23 boy star see felt embarras think girl feel
lot comment endur even hour near place current live
harass even
thief snatch gold chain cut throat admit hospit
survey carri safec red dot foundat along safec audit street market mumbai
man tri grope
cook go fetch water school found male student touch femal privat part
man sexual harass
boy touch privat part
girl stand bu munirka person fall touch happen even
call samch lay hey cuti chocho doll nomo deymo hey pretti girl
street light return home templ way tea stall boy smoke saw start talk loudli laugh
girl gangrap sever boy along river kibera south dispensari plantat along river
afternoon teacher got class start insult call sort name thief lair stole book belong school done infront classmat anoth student art depart realli emabarass
heard girl face problem like snatch stare bandra east station
return school two boy whistl teas
market even time deliber push
friend travel bu crowd man start push friend purpos touch privat part start shout man got scare left bu
comment
visit gb road area near new delhi railway station lot pervert friend suggest pay visit roam casual initi saw 23 polic check post primari school puzzl realli biggest red light area delhi heard walk swiftli clear confus attend broker mistakenli intent interpret potenti custom quit surpris see fear face promot illeg act prostitut went ahead explor situat sneak insid 100 brothel depress notic girl less 12 year rais skirt lure curiou know start ask question booki monetari aspect e charg complet taken aback bylisten offer twa pretti much around 200 buck orgasm singl shot call around 800 1 full night even half expect still curiou end littl journey almost hell enquir insid detail experi pathet youngster leav brothel number hundr twa aw hear procedur choos among hundr prostitut insid brothel chosen money paid advanc cashier receipt provid along number turn go wait balconi offer almost everi thing would like drink eat horni youngster also advis us visit brothel 64 other poor reput loot peopl job done clear violat human right prostitut job around 1617 hour sorri write less time right would quit happi although mail detail write soon first time got platform express concern keep
wife way back home work car left road wait traffic move car 4 men tri cut take left without follow road rule wife allow take left littl spot happen car 4 men stop car glare car drive
resid chennai observ mani incid public buse women touch intent fall women driver appli brake tri hold women hand rub back mani incid disturb women physic mental almost school colleg offic women affect note idiot men includ 1 even bu driver conductor mind set good respect women 2 employe colleg guy high school boys3 elder uncl old age men 50 age men 4 gener men age 10 80yr involv stupid activ crowd buse mostli women colleg offic area present chennai 27h 24c 29c 54 seri 54 buse ponamalle porurdlf gunidi saidapet note pleas add crowd unsaf area transport facil bu number share auto train cycl stand bu stop railway station cab etc comment report avoid atleast women harass women even good men rais voic worst incid even make govt take new action safeti women govt might arrang extra buse crowd area peak time special women buse women conductor women special share auto cab women polic near women colleg public place daili check crowd buse women polic
friend come back school read harass way boy smoke tri rape attempt run caught put groud brutal rape collaps rush hospit
someon comment friend kya maal lag rhi h even
job hunt man promis get job gave phone number ask call follow come meet place got ask sex promis give job money sex
walk metro park rohini east metro station afternoon guy made indec gestur open zip report secur staff metro station
return tution class two boy bike came hit chest shout
physic abus store
wait sister metro station boy stand around start ask sexual favor
comment filthi stare impos upon happen night 900 pm
boy lookik girl pass barber shop whne girl look embarras boy start catcll whistl comment boy harrass girl caught look th incid wink turn walk
dtc bu morn ask guy let sit ladi seat start pass lewd comment even driver conductor said anyth
drunk came friend way colleg start comment inppropri
go sinc way school lot time face stare whistl etc
fianc time wife readi return bike person work unknown man pass hand back deliber manner behind went past decid follow person get dark even though fiance angri
travel local train leav place middl age man forc touch privat part slam badli
boy age around 10 12 year touch breast walk pavement shout escap
survey carri safec red dot foundat along safec audit street market mumbai
man tri snatch chain outsid sunday market women around came rescu start beat call polic
gone visit friend dress skin tight top pass young man call respond gather around tore tight kind woman wrap lesso terribl
comment pass contin happen 17 05 2013 even
go bkc station guy cmomment
comment indec exposur touch grope happen even bu srcc place bu guy came happen
guy look masturb near metro station entri first sure start groan saw
drunk men car follow around road
two guy secretli take friend pictur also blow whistl
night street vendor pass cheap comment
afternoon go back home school notic boy indec expos stare girl
subject lewd behaviour
went late come back attack three boy due short expos dress put boy pressur rape
wait bu guy pass cheap comment
ladi chain snatch buy someth street vendor
travel bu restaur nearbi notic certain group men stare openli none interven stop felt powerless
buse almost everyday face vulgar andd inappropri situat
catcal whistl even
ogl facial express comment
night wait bu sudnelli phone lost
comment market
alway certain catcal whistl dtc bu incid went happen afternoon
travel bu ill treatedbi man
walk school met boy sit infront video den start cale went start tiuch tele beauti need watch movi
harres guy spoke vulgar word
touch grope
crowd bu guy sit seat reserv girl girl ask leav seat polit guy gave set start make face comment happen even
man tri touch
walk across road go board matatu met straanger start tell wish could kiss lip deepli felt uncomfort walk away
 82% |###########################################################             |
touch grope amp hit
harass
cross road mother 2 guy pass comment wait bike gave us look went away
even phone stolen
street light avail point road near open ground sec3 vasundhara point id depriv streetlight unlik type point area moreov n nubmer poor resid slum set local make condit even pathet area get dark creepi night due frequent accid unwant incid take place poor live highli affect alway live fear risk mishappen take place point time night
wait metro kashmiri gate group guy start comment behav weird manner embarrass
happen ridg near north campu morn man indec expos buck group femal runner morn
happen friend gym trainer work gym
sexual invit girl wait cab pickup offic hour mg road indec coment amp play laser light
seat ground dc offic group f men also sat stare passerbi make differ utter especi ladi
two men tri share rickshaw said would like sit exit men start say would like sit near thank auto driver allow board auto
comment touch amp bad facial express way back dwarka sec1 dtc bu 764 727
harass
ladi worn short top pass men sell kitchen item start start use funni languag describ
metro station wait auto riksha guy kept stare make weird face
misbehav
harass vasant kunj
truck driver touch tri sexual harrass mira road wit harass almost everyday happen highway opposit thakur mall dahisar highway mira rd
catcal whistl afternoon hour
fren mine got follow boy walk alon street two guy follow stay shop smtm cool situat guy wait outsid shop ran enter unknown person home nearbi shop enter hous guy stop follow
friend follow someon even
9year old school girl shadrack kimalel found injur bleed behind lindi friend church lindi villag defil left scene
sent shop night buy salt met aman touch butt
boy bike snatch chain ladi walk near market
stuck finger perpetr insult lewd remark dirti gestur punch report polic
made inappropri express happen even
come bank man brush past touch almost everywher pretent space bank guard look happen without say anyth man
touch grope bad manner
stand bu stop man click pictur also auto wallah tri take wierd place
follow bandstand make awkward hand gestur forc tri strike convers
friend follow metro station hous day week
happen around 7 pm return class person came ask friend seen person follow sever time
class eight boy touch girl privat part
unsaf area prabhadevi market
walk street two guy bike stop front scare laugh abus went away
night street light make area realli unsaf
realli bad
girl 14 rape school campu attend school parti night parent wish ariv attck four guy ask money forc rape
two boy comment pass happen afternoon let dog bark
hblock shastri nagar 56 yr old small girl rape shopkeep
come back offic around 830 guy follow iifcp chowk reach kapashera border auto start show genit
saw guy comment girlfriend figur walk
group boy first stalk victim get opportun made comment embarras
incid happen return school men stare make awkward facial express incid scare
guy tri touch
went school camp 10 std visit mapusa market us bu realis man unzip pant tri harass shout ran away inform teacher
incid occur aunt famili shop mall found boy tri take inappropri pic
comment mani men outsid movi hall left without watch movi
misbehav
travel alon return select citi walk two boy came toward start touch hand
even travel public vehicl crowd time person back put hand back couldnt see vehicl realli full
face disgust comment catcal multipl time
vo bandi ji k sath hua vo evng k tym pe waha k flyover k nich se aa rahi thi n tabhi kisi band ne phich se attack kiya n vo ladki chilayi bht zor se tab vaha luckili kuch street pe rahn wale log ane alg tab tak dheka vo bhag gaya
area get pretti lone night men come smoke keep stare anyon walk
man 40 year defil ten year old girl convinc normal feel nice touch privat part
walk guy keep touch behind uncomfort
sister insan man take advantag situat rape hiv
even continu comment
happen girl crowd bu touch inappropri
guy stare continu
went walk friend near home person came bike tri touch wrong place
took place afternoon group guy stare gestur rude vulgar express
sexual invit pass lure comment
walk friend around 910 p north main road us cover upper bodi scarf sinc cold bike slow touch friend breast acceler stop bit ahead look victim face gross enough catch even though tri start shout
whistl walk alon even
survey carri safec red dot foundat along safeti audit street market mumbai
alon road man stand road stare afraid
eve teas blue line metro rajouri garden
return home school boy street stare start comment whistl also make wire facial express scari
wait bu busstop two peopl stare
bike rickshaw go back home look comment badli
month ago boy block way return home whistl
school walk class boy grab hand stop told amaz bodi would pay
usual travel night metro amp guy keep coment amp start whistl ignor
pinch breast gover offic
guy stop car comment
misbehav
incid took place govindpuri kalkaji lane 5 market afternoon wherein notic boy harass girl took hand start sing song also use inappropri word boy question boy start beat month
waj happen friend travel bu
two girl travel rickshaw two boy bike whistl look happen even
liftman look elev didnt like way star
age 16 yearstyp harassmentstalk choglamsar leh even catcal choglamsar leh market school leh eveningday time comment school leh market time touch school time sexual invit choglamsar even other ask phone number time
student sometim feel safe
incid took place metro station afternoon go home boy start comment
man beat wife children watch quit disturb scene children
walk 200 mt stay 9 pm dollar layout j p nagar 4th phase man bike wear helmet hit behind spank grope chest sped sinc rain could catch bike number properli make similarli alway guy around stuff like local sinc roommat face similar problem 8 pm men pass snide chep coment u walk street even 6 pm u visit superstor friend told polic van round secur still lax someth done feel unsaf step dark know happen grope might get wors
happen even
peopl drunk tri molest peopl slum area mainli youth 6 number shockingli teen well pleas stop unless mani us would even think twice go home
heard news girl rape knew girl known harass went quit share anyon final came got rape room uncl depress news
friend mine sexual harass guy quarter rape simpli guy ask refus guy emabrass mock friend get girl plan friend rape girl badli injur almost die
bhayandar station rel tri touch inappropri
friend brother visit us hous home sat usual start talk next start touch tri resist start pet luckili younger brother came left
comment
share autorickshaw person sit next tri elbow stomach later tri slide hand thigh told take hand away sit front
aunti rickshaw chain snatch two guy bike tri chase car unabl
toi market hawker grab waist rest laugh noth click
friend went occas tipsi drunk plan either directli indirectli way drunk immedi commit sin rape
mother follow sexual harass man took dog walk
catcal afternoon kashmer gate metro station
boy harss quarter thiev 500 000 fr parifoot game cours harass almost kill
night went see movi drunk peopl mani roadsid vendor comment tri touch annoy ask policeman bu stop
husband beat infront peopl refus sleep morn also abus badli touch privat part infront peopl stress embarras lot
man tri becom close insid public bu even kept hand thigh
group women head home girlfriend beam whisper made move toward poorli dress open back dress
much teas push gang boy earli morn
go class way park group boy let go ahead pass dirti comment whistl
sister follow two boy return home school whistl spoke bad word
afternoon hour underw comment
 83% |###########################################################             |
drunk guy tri harass move local train
gone visit cousin stay night arriv area follow disappear bent
boy abus girl
night boy comment
friend mine sexual harass beach
friend harass even around 6 month back someon held hand
night low street light boy whistl follow
return shop alon subway find group biker way feel scare
girl know rape street murga chowk stop go area alon
survey carri safec red dot foundat along safec audit street market mumbai
teacher harass girl touch privat part
guy make inaapropr express
man tri touch weirdli shout ran away
boy drunk start misbehav
happen school girl return home school crowd bu men tri touch privat part
girl walk two boy follow start teas cri felt helpless boy stop teas
4 us wait bu stop particular bu someth irk someon watch us turn side see man show privat part smile blood ran insid much anger told cousi elder said avoid helpless public man went shame ful work immedi bu came board
misbehav boy
comment
walk way saw two boy whistl comment girl walk alon street
touch grope
go back home offic person tri touch arm
girl teas felt uncomfort
men sing vulgar song stare
saw boy teas girl make outfit later made inappropri comment
ogl morn ina bu stand
4 member use go offic mmt bharath nagar hitech citi 6 15 idiot indec fellow get ladi compart start indec exposur problem solv b coz polic protect scr
touch school boy
go institut bu suddenli boy start touch teas back first ignor limit object incid happen mani time bu
place way mahila colleg girl high school girl face problem cross place ashram chowk
uncl want kiss promis buy anyth need
man star stand bu stop
incid took place even around 630pm guy start follow mt sister got colleg bu pass lewd comment
walk street 2 guy cycl ride pass whistl start sing song cousin sister walk road
comment
follow man bike walk home bu stop
harass kurla station
come school gyan kendra submit project around noon left start walk back hous guy constantli stare till reach place enter wrong build stood 5 min would go go back real destin
come back station guy use sit specif spot whenev girl pass use urin face toward girl
sexual invit
comput teacher touch inappropri happen often
comment group guy near gk 1 even friend
stare
went momo realli bad experi guy scold girl badli cheap word
cheap comment bodi pass
3 friend come home way teas follow till home
chain snatch happen eve teas
happen everi alway group boy hang local call name sing sometim follow us till bustop
purs snatch biker
friend shukar bazaar shop man approach friend start forc go shopkeep came rescu happen afternoon peopl stood laugh us told us girl deserv
travel tokha visit aunt stale middl age man seem weird alchohol
travel rickshaw notic rick wala glare disturb
happen afternoon
go home finish class boy pass comment whistl pass
stare dirti way well comment pass
247 2 men bike catcal
man leer get back tution late dark never forget dirti stare
drunk man tri touch breast nowher riksha
first bangalor spend 4 year delhi kitchen yet set henc male friend went dinner restaur locat central area 1030pm done dinner head toward car park 500 meter restaur four men look like forti start call say excus turn toward start teas pass comment like miss hello friend got furiou somehow manag convinc get car chose react howev friend ask stay insid car went note car number men seem notic note car number start man handl fortun naga chilli spray took ran toward manag knock three fourth man got hold hand push road hit head saddest part peopl around watch drama play road nobodi came help friend busi end tri fight rest saw men take iron rod car dickey pull friend tell gonna come help abl help either ran toward car immedi went insid turn car take uturn two three came run toward us intent hit us sinc alreadi insid smash side car glass rush nearest polic station ask help show car number friend luckili note polic personnel hesit file fir per understand seem local polit goon resid area left car polic station night return home 3am morn went back polic station file fir repeat fail request use friend posit armi spoke head polic station file fir total devast insid want leav bangalor cri whole night ask friend drop home mayb month two mayb forev woke next decid fight woman live alon citi differ look car peopl easili identifi want get mess make round polic station court case friend also leav work day decid take back fir also stay back bangalor move look new hous move farther safer hous may question someon head withdrew fir tri punish culprit know countri live even murder set free punish leav asid eve teaser peopl made injuri us physic damag properti stori want share show countri fail protect citizen stori faith woman countri govern author stori show human lag help hand other need ye famili watch incid stori tell matter much hatr world put still need love human stori hope women shall overcom pain
nainit famili man sit bench pull chain show intern part
incid took place hotel p g deep sleep sudden woke earli morn 630 caught work male staff room pull night suit call owner get hostel
sent shop buy cigarett wehen man came touch breast shop
man whistl follow
entir stretch littl ahead movi time theatr ramchandra lane extens unsaf due absenc street light entir road dark fill pothol ad presenc liquor shop cigaratt shop lane attract young boy sit drink till late night lane also slum settlement make thing wors slum dweller add unsafeti road
street boy harrass neighbour night buy veget
walk pass railway bridg kambi muru man whistl twice respond shout quotth ladi purpl nice leg admir wifequot
touch bad intent
harass happen everywher
cross satya bridg boy call masturb sexual invit
return colleg old guy whistl near poolchowk facial express unus
boy came ask money refus start touch leg block way
comment girl afternoon
harass borivali
age 15 yearscatcallswhistl leh market
poor street light
girl molest boy
school bu friend went submit assign teacher school use school bu go home conductor use bad word start teas us complain teacher
way home boy bike tri touch follow tri snatch purs
sister stalk two boy way tuition even street light work could see scare drop
molest
group guy comment hoot
incid took place night around 6 month back muharram man touch friend privat part went away happen front mosqu
today got crowd bu man tri grope kept think react situat
sexual invit 8th feb 2014 even
car follow girl invit illicit comment indec act
ride scooti two men two bike follow scooti teas disturb ride
stand bu stand saw two guy click pictur
group men seat outsid bar call young woman quotdarlingquot pass girl tshirt written darl
group guy tri touch inappropri way
chain snatch resid delhi polic appart
ass pinch
travel train famili gener compart place sit guy besid start touch waist tickl first thought shout switch place dad happen local train gener compart mumbai
man came closer touch thigh slowli travel bu suddenli chang seat
incid occur year ago face number harass affect thought share let peopl bewar friend girl trip local medic colleg exhibit return good overcrowd bu guy start eveteas sort comment manag get sit next corner allow bu conductor come take ticket happen insist know name good friend doctor faculti premier institut reveal name detail till date fail understand whether understand naiv purpos competitor school ultim tortur came end quick stop polic station friend laugh ask justif action time pass mother tell kid bewar human predat world warn friend tri caus harm anybodi read messag bewar girl friend also
ladkiyo pe comment pass kiya jata h ladkiyo k travel problem hoti h
morn come bu uncl put hand lap ladi ladi turn red becoz shame could take action
sister botan garden friend two guy propos forc accept propos
touch grope
boy toouch comment girl
touch girl walk along road
man bike hit butt fled
person snatch bag return coach
train boy pass cheap comment
boyfriend lie sex stand would make pregnant refus forc
men took picur without consent
incid happen outsid bel township friend bike talk colleg friend outsid home bel township taxi stop pearl white tata indica vista ka 03 aa 5562 driver start teas friend start abus us look intend pick fight road chose quietli note vehicl number report two other car incid went ahead toward yeshwanthpur proceed toward metro
take pictur vikaspuri
gang boy loiter outsid station platform pass comment
 84% |############################################################            |
go new baneshwor touch unnecessari stranger
happen girl drop bu reach call two boy come greet boy boy ask greet boy tell girl privat part disappear greet boy follow boy beg girl becam afraid boy ask surend give possess money phone chair bangl collect girl start cri help peopl could come rescu boy excap
neighbor famili husband beat wife
sure rumour true institut girl accus professor molest return id card back say snatch id card back ran inform friend famili complain
girl walk district park saw somebodi click pictur treid hide face boy still captur 2 3 pictur complain old man sit park phone taken away uncl hit hard
elderli woman advanc sexual young man refus report polic fals claim harras
morn hour wit ladi chain snatch guy bike report polic
boy keep follow night ask escort shop
usual back hostel local guy call chines
class boy tri touch inappropri
use medic like himalaya product like muscl relax oil ayurveda privat part reveng 1995 mother brother big issu finger auto stand
minor person best friend rape invit hous 15
comment
bu happen
feel sexual harass incid almost 4 month back walk alley boy teas comment bad rough word even stalk stare touch scari moment never could forget final manag escap hell
follow men
guy show peni bu stop utter word scare amp kept eagerli wait bu come place desert
go tution class guy stand road stare pass bad comment
harass
facial express touch etc gang boy
unsaf hotel mussori attract peopl cheap rate harras
occas like jatrasindec exposur touch especi grope happen woman mass peopl bad intent
follow guy
walk school certain man told beauti want marri futur
touch
walk road 2 boy start comment throw chit phone number written
campagin remov black film vehicl forgotten alert govt public use black film curtain buse etc
stalk walk alon changspa street
ogl comment
woman drunk late night scream help peopl assum think drunk actual rape 6men
bag wallet snatch even time
whenev go somewher decent dress peopl give whistl feel embarassesd
man caught wife cheat got hold wife beat leav bleed take woman
block way make suggest beckonng gestur put head walk around beaus way larger stronger cross road stop debat safe distanc report uniform exoect threat doubli threaten cant think solut said fail
realli bad
harass afternoon
grope
comment catcal whistl
old man tri touch friend final rebuk person happen morn hour
stare like never seen girl
man touch inappropri bu
touch grope public buse
guy persist chat refus advanc way back school met guy mile 2 nkwen bamenda stop said need talk realli tire need go back home held forc arm rufus let go pleas avail let go final saw two male friend eat call guy grab freed left two male friend
even indec touch grope
sit window seat bu boy thrown balloon
stand sister bu stop wait bu two guy came car actual invit us car lewd comment
return home sister home face gang boy start whistl comment stare took way home rush
go buy veget near main road petrol pump cross area truck park men drunk misbehav tri grope scream peopl came rescu
railway station 2 year ago crowd uncl touch behind
mela mount mari man approach behind touch wrongli said crowd happen mistak lie
wait bu way colleg man came close touch breast start say breast beauti
walk railway station man brush lot space walk
3 boy follow way colleg
incid took place even girl walk man suddenli came construct barrier metro hhe tri hold sinc red light mani car abl
walk saw someon stalk
late friend home home way boy threw pebbl teas
street light
incid took place 21st novemb 2012 around 6 even two men stand post whistl pass comment
friend travel metro stranger held hand
teas boy
guy comment make inappropri express insid metro
comment catcal whistl ogl facial express
two three guy white colord activa follow friend rickshaw tri snatch phn friend tri save nd phn run away ws realli danger might weapon pleas look matter asap
ladi ask chang matatu tout refus give back money arguedthat ladi didnt knw work shouldnot start teach job took intervent passeng woman get back money
school day mental tortur threaten
young boy took pictur walk
guy whistl call get physic involv
morn 2 men beat 1314 year old boy badli cycl get touch bike
walk cafe along friend guy start sing tacki bollywood song point toward us
harass
old man tri touch young girl
head toward colleg marri man sat touch sensit organ bodi initi ignor action repeat ask stop
bo friend alway stalk pass comment quotbhabhiquot realli annoy
teacher beat student
friend wait bu stop man stand know man give discomfort facial express could toler friend could anyth left
harass
liter less block hous 7pm walk slightli slower normal attract coupl look young guy came behind didnt see come slap grab butt sprint away opposit direct jar especi cuz 7pm close hous blink eye mainli lower incom men close time shiver
friend stalk 2 men day stretch told us scare confront keep ignor told react thing go hand
comment even
grope inappropri comment
random sardar metro tri molest
two boy bike snatch chain girl uttam nagar metro station incid took place 1st septemb 2013
walk home truck pass slowli side guy insid truck made weird face whistl comment quotaunt get wayquot
take pictur girl pass abus comment
two young ladi cross road toi market way toward kibra law court men comment said quotthes one need penisquot
comment stare area even
walk lone street guy make filthi gestur invit sexual happen night
even
boy pass comment chinki look
morn metro man constantli look ogl facial express
creepi peopl tri follow dark
catcal comment bad facial express take pictur indec exposur sexual invit
even pictur taken confront guy delet
polic offic comment
ladi cycl rickshaw ask boy direct snatch chain
come back visit friend place near dana paani beech boy start whistl sing song made highli uncomfort
incid happen taj land end coffe shop meet founder save life foundat even midst seriou discuss man came us first wish colleagu turn said someth effect guy fking real good could react disappear call waiter ask thing could happen 5 star hotel assum quotsafequot space claim drunk hang around convers hotel question find procedur preemptiv measur patron drunk well find ident guy rais question need alert time street harass seem move indoor could seriou person decid physic assault access cutleri
guy dtc bu intent touch
grope bandra station old man seem intent
chain snatch incid coupl guy bike snatch necklac woman stand across street
visit place see rel night unsaf
aboy forc kiss peni refus ran
tri get rick ghansoli gaon becom difficult around 7pm sinc gaon peopl would roam especi men make woan uncomfort
scooti alon two boy motorbik grope start laugh scream drove fast could shock could happen aligarh could note bike number
go toi market came across man hit shoulder rais eyebrow wink piss
senior came us second colleg group start abus us without reason
girl harrass group man abl say anyth
guy misbehav
stalk drive alongsid ask person info like name home etc
 86% |##############################################################          |
two biker snatch chain ladi near jnakpuri metro station happen morn
felt unnecessari touch around privat organ sever time public vehicl area feel uneasi public vehicl
night sleep got phone call around 1230 pm got frighten receiv call think might sort emerg urgent call receiv call came know bluff call man talk want know hung phe continu call n disturb whole night
face friend due work bad experi
teacher bad intens student touch us sit bench
two men park lot kept stare go toward car kept ogl made realli uncomfort glare back giggl
sexual invit hoot
girl rel uncl night girl uncl touch girl asleep small take action meantim mother enter room saw activ done law care societi fill complain way save daughter
teas go school come back school local boy bhatbhateni area
name conduct survey guy touch girl indec manner north campu outsid metro station
comment khayala new delhi18
girl boyfriend continu cri guy hurt hand badli happen even
ogl travel bu
mani incid experienc want write happen year ago girl 45 year rape neighbor parent took hospit take treatment
male daili travel use take advantag crowd bridg use touch press ladi
follow someon
phone got stolen delhi metro night
ogl
born brought cochin experienc mani differ sort experi rang comment indec exposur touch privat buse run citi last 2 row ladi seat ladi tend sit reason old man age around 6065 use follow show indec bodi part everi school place till home area mom use live alon scare even report polic
afternoon girl 6 year pase call boy start touch
stare comment
touch man way back home tuition incid happen 9th grade minut hous incid took place randomli smile stop touch time stood still
happen morn time walk girl touch inappropri person ran away girl reaction
verbal abus indec languag behaviour teenag boy
comment
waterfal near karjat drunk peopl pass comment stare friend
young man follow come dhungedhara bottl bag catch hand pool toward coundnt anyth big enough fight ppl around help final shout left
colleagu also senior work kept forc take lift despit refus kept call text follow work stop confront told interest complain manag fine
misbehav
read news papper read news wear scream news girl rape 3 peopl read news n fill sream
feel elpless
walk cousin road mani men stare small sister around 20 ignor walk along act like never happen
man start teas bu way school
touch grope sexual invit pictur taken even afternoon
girl boy talk among say chines though indian
misbehav
34 boy shout friend train platform inappropri manner
friend mine group ill manner guy comment obscen bodi
sufferd month wrong caller badli take either follow ask whu u saya di compani number u trace u pl help iwa sufferd person talk c campkurnoolregardskanuradha
way back home met boy whistl ignor start insult
employe taj krishna husband stay 1 week wear loos fulli cover cloth spite
public vehicl touch bodi part boy
friend goin home school alon boy stand street comment badli realli ridicul
harass guy whistl pass loos comment
apart stay friend morn guy sit next friend shock moment start touch scream sinc scare sleep alon person escap knew structur apart well
boy 1213 year old pass filthi comment friend
irrit behavior even
walk friend us teas
take pictur girl pass area
travel bu alon boy sit besid tri touch feel insecur scare
catcal comment even
guy walk behind singl sexual explicit song
walk street saw group boy chit chat laugh cross start call name sing gain attent
famili function home harass want intimid
friend stalk 3 men car
studi spohia colleg suppos enter colleg pedder road gate didnt id card henc go pedder road main gate walk man bike wear helmet didnt see face pass friend toward side said like bodi wear decent long indian kurti comment realli felt disgust couldnt react because shock incid happen first time
alon home later father came offic forc touch breast cri lot
happen 6 month ago go fetch water even boy even know call say type nonsens concern
guy make indec facial expressin
group girl pass school certain boy start call girl never look back nad boy start abus shapeless
return colleg group boy teas lot say bad word
month ago group girl wear go fo hike shivpuri three gal jugl group boy wear thre n tri whisl n sing song us ran away
ladi faculti art ask date lectur could pass cours ladi told
whistel catcal grope becom frequent thing proper light unsaf women
girl almost rape birth parti due intervent girl rescu
even guy pass comment
auto driver even pedestrian stand occupi whole skywalk intent grope girl women tri pass strict action must taken control imbecil insol freak think behav pleas women forget woman gave birth
sit domino wait friend group boy sit tabl next start behav inappropri stare comment
girl neighborhood rape park near kali basti lock parent know happen
touch ogl etc
stand bu stop anoth girl stand behind suddenli maniac man came tri kiss whatev dont know cut girl lip
catcal
two boy stand center market stare friend
go back tution around 2 pm auto driver look pass comment misbehav happen near virat cinema
travel bu guy grope indec constantli comment
incid sexual assault
comment munirka
incid happen night june 2013 young boy around 1415 year old take pictur girl back
street light comment
come colleg boy stand street stare pass comment laugh
walk look butock say quottumetosha mbogaquot
man kept stare uttam nagar east metro station
maintain hous person abus us expos take cloth hit door hammer replac door new drunken got anger happen ask go hospit treatment ill person need take treatment
man use repeatedli call tri make convers ran away stop
man stare badli bu tri close
return home 730 pm public place group men catcal peopl around ignor situat ignor name move
ill treat
guy flash near park
saw taxi driver drive stare ladi pass stare top bottom liter stare till time visibl sukh sagar bu stop
way uncl home saturday man follow young girl whistl
group boy whistl go toward home
saw group boy whistgl take pictur girl go school
colleg friend wait bu came old man invit sex ignor went place mani peopl
shop tnagar road chennai notic men call name disturb
chain snatch
realli bad
man tri rub elbow breast
follow group men car way home
train gent compart amp guy tri touch inapproprait
lane commun grope
take pictur continu comment afternoon hour
auto walla wass make inappropri express
boy follow friend gina way boy said love much keep follow gina
catcal
comment happen night
saw 1 guy take pictur 23 girl
wit incid guy tri touch girl travel bu
certain woman call offer lift whe enter car drove unknown place
shivaratri street full crowd crowd guy pull touch back portion friend bodi time pinch home went away
follow metro station
realli bad
gener touch grope stare metro station comment catcal street
number 919231688222 guy call odd hour amp send lewd messagesi bangalor ye guy calcutta send messag bengali call odd time send lewd sm
survey carri safec red dot foundat along safec audit street market mumbai
north east report
grope man also forc upon woman
touch stranger birla planetarium nake man expos friend along ajc bose road
men follow girl skywalk bandra station
sexual invit
wait metro afternoon two guy take pictur could say anyth unabl speak hindi
ek baar main famili ke sath mahalaxmi ja rahi thi ek admin mere brest pe hath mara tab main bahut dar gayi thi aur gharvalonko bhi nahi bata payi
 88% |###############################################################         |
peeopl touch also comment
girl rape mile 14 dibanda got pregnant realis man involv cult
guy use comment sister way tuition
ill treat
boy stare
boy usual roam around place pass bad comment girl whistl etc
dtc bu cpchandni chowk experienc indec exposur touch prod
indesc exposur total nake man ladi compart strip insid compart
boy stare friend
two guy motor cycl snatch chain ladi get rickshaw
stand around 8 pm bu stand guy stand comment ignor
guy continu comment girl metro gener coach afternoon hour
men comment friend
boy chew khat whistl way home school
use cale name left school home name like beauti
girl pass buy near group men men start call girl never paid attent men told cheap girl
neighbor abus child three month scare could tell anybodi
market friend boy stare us start comment etc
notic group boy alway sit along railway line wait pass call name whistl
afternoon yamuna bank station lot touch
group boy use teas friend pass differ comment
peopl road haress colleg girl
child abus
wa travel colleg bu person com near touch bodi part
school teacher call offic offic start touch breast forc kiss
septemb 2013 incid bad comment boy happen maharani bagh
go colleg touch hand
friend sexual molest cousin 12
father girl die mum got marri anoth man step father like girl forc go stay aunt aunt return gave girl work anoth woman aunt receiv payment
watch porn quotveteran video roomquot 8 pm till 10 pm
indec exposur comment
drive twowheel uncl pass scooter nearbi comment give blow job
man look badli give bad express
seen girl stare upon guy make feel uncomfort also whistl ogl
get back home school man approach ice cream told take took start touch wrongli threw ice cream ran away
go baluwatar balkumari nepal yatayat bu pack saw ladi enter bu anamnagar man around 4045 age tocuh ladi ladi felt uncomfort time everybodi start look man felt bad got bu
rickshaw back tri touch
eve teas comment
boy wdho sit soweto play ludo alway make bad comment girl women pass embarras much especiaali someon respect
realli terribl
travel alon comment chase peopl
nerul station daughter climb fob pervert came stair expos target daughter
travel kashmer gate bridg touch call vulgar name made sure never ever travel like
bhilwara labour coloni visit apart hunt labour bulli group show us knife side road
take pictur pass lewd comment
happen near colleg vocat studi sexual invit guy pee public
travel imadol due work guy knowingli touch breast
think area nearbi safe night girl
girl aunt walk road two guy came bike snatch chain thu injur
incid satya niketan foot bridg person comment dress sens girl
mid summer season friday night friend come back night virgil someon stop refus stop suddenli boy came bush circl around lateron rape around three guy
harass
harass
follow tail partial bald middleag man wear armi shirt might polic sepoy purpos jog behind stop sli part even confirm ed stop randomli
drunk rikshaw puller comment
shop shopkeep said maal got angri slap guy happen 4th may 2012 even
man masturb banana plantat look tourist walk road
friend walk home group boy seat around corner start whistl us pretend heard start call us name felt realli bad
happen girl stand next group boy whistl comment
two guy take pictur whistl femal friend
2 person bike suddenli came behind push rel snatch chain mother neck
walk street way work pass mani peopl shock man put hand crotch 11am although know isnt best neighborhood walk everyday must notic wasnt pay attent went male cowork well happen fast crowd peopl noth could
walk home stage man know kept shout name tell smart boast kept tell look shout shout ignor
walk street man came an ask night stand word made numb clueless approach push hard ran kept run till reach public area
accompani 3 friend go somewher man stop car ask direct give direct pull pant show peni
touch indec exposur
poor street light
travel metro saw man tri touch woman inappropri ignor could interven got next station
street light mani part
oogl also even
chain snatch thief
travel via mirco bu even amp man sit next start whistl make nonsens express
incid took place around morn basic incid happen commut metro toward karol bagh inappropri touch fulli load metro
touch
2 boy follow kamla nagar till hostel whistl told go away made even abus ignor walk faster
person repetedli dong
ye sexual harass sever time life
go friend hous wear nice chain two peopl bike suddenli came tri take away chain besid full effort tri save chain
walk roadsid man whistl look show tongu click continu walk
shopkeep use misbehav girl tell lie help fill lic form would invit home touch inappropri
catcal touch grope even
afternoon middl age man constantli stare leg stop even though ask eventu get next station
night guy continu follow
even man tri touch woman bu
report taken dadar street market exact locat bigg sale societi store interviewe feel safe place due rush
friend mine 15 year old suffer molest elder confin hous neighbourhood whenev use come home alon school
harass marri man
brother friend slept hous night tri rape succeed smart
auto driver tri harass friend number eve teaser increas
abus happen abus friend school campu sat togeth discuss two thing discuss said someth made becom angri live quickli made feel bad
guy pass hideou comment girl dress pass nearbi took place even metro cp
friend cross road random guy signal sit bike call us thought need help came know fulfil pathet desir
pass street men start pass vulgar comment
follow unknown person return home realli scari
peopl touch also comment
ogl touch grope bu rout 392
holiday went back home see parent alon home unlc paid visit sudden start harass shout help
kamalakshi came outsid colleg around 1 2 pm head toward atm though atm near colleg work go khandagiri yet cross ghatikia petrol pump suddenli kamal slow becam bit unbalanc got scooti road time bolero cross us person shoulder length hair sit window stretch hand air later kamal told ride saw mirror person took sperm pube tri wipe got road
happen metro even night
come back colleg group boy start comment make vulgar express
saw certain girl two boy touch breast
wait inderlok metro station exit gate group four boy pass cheap comment follow see policeman turn around ran away
breast bottom grab press ladi get buse also molest come realli close stand behind ladi touch differ place ladi tri get bu 6 830 pm bu stand crowd buse also normal get crowd grab bodi part second press vanish crowd pleas help
cab happen mad man pull trouser
guy use follow month februari use travel pitampura coach class
friend teas three man whistl us well made strang sound use tongu
guy comment stalk
man stare even time way home sister
friend friend wear short skirt crow place guy touch thigh
way schoool mate boy smock grab friend start touch breast told run look back
friend return friend home boy chase us ask name phone number
 89% |################################################################        |
go home school lone even group grop boy n whistl n sing n ask name
travel rickshaw man tri grope breast
policeman misbehav state bank india took place 5th octob 1115
comment
call girl come along could identifi bike helmet age around 35 could ask help around less pedestrian movement around 345 p rais voic left
catcal
age 15 yearstyp harassmentfaci express school leh 2016 time take pictur leh market 2015 time catcal school leh 2016 time
even somebodi comment
travel train nagpur chennai men kept follow stalk regular interv
wait cab boy continu star
girl came school met man kidnap end rape
boy sit far away friend kept say someth girl haiderpur villag stand balconi
call name villag zanskar
happen even
friend teas man handl station bridg cross way back home night way back home colleg near mulund help ran away didnt speak day later spoke
go colleg boy pass comment us
middl age mad start masturb stare sinc first time happen yell loud ran away men gather tell bambaiyya languag quotwoh hila raha thaquot understood happen
happen friend tution teacher harras long time
0300 pm time govern colleg get student take buse go back home time peopl correctli teenag labor tri harass school colleg girl either comment look whistl girl hour avoid take public buse instead carpool classmat walk major way back home govern taken seriou action group girl complain incid nearbi polic station
decemb 2012 februari 2013 travel around india lone white femal two separ occas two differ hotel nagpur group men attempt break hotel room clearli intent sexual assault terrifi spent long wake night furnitur move door bolt drawn stop get men persist came repeatedli night tri forc door open could open door forc would call ask open door sometim said work hotel need come recept 1am point fire tri save claim four previou attempt throughout night forc door open even knock could hear 3 5 differ men outsid room tri get
two boy block way enter bu also teas badli
comment sexual invit
stalk creep becam 13 loiter around hous follow around wherev went groceri errand touch inappropri public scare went time final told mom follow went slap got public thrash never saw
chain snatch rob jewelleri
harass train
friend came back home night guy outsid bar panvel comment whistl
friend walk road two guy bike slow next us grab friend breast sped even get time react complain crowd space yet dare
teacher rape pupil
stalk
alon delhi wait driver come pick almost 11 pm 4 men around call brother home could get driver phone talk person sent driver man start walk toward scare death could sens fear brother tone pass driver came run could say enough thank god
happen 2nd std build lift grope
peopl found comment women lat night near raksham nilayam
incid took place 29th septemb 2013 around 8pm khayala near pacif mall pass area saw boy whistl comment girl pass
saw two boy bike snatch chain ladi go walk happen januari 2013 morn
click pictur without permiss
way school mate boy forc stop talk
threaten touch indec exposur night
tie four friend went see movi grand masti boy sit back make sexual voic distur also commentng att
wit chain robberi
peopl near old fort near old delhi metro station bad activ
landloard wife routin drink silli go nake call girl
place sector 16 vashi navi mumbai 10 30 pm got sector 16 bu stop best 525 bu start walk toward home footpath shadi without street light decid walk along road walk someon loudli scream footpath quota idhar idhar dekhquot involuntarili turn direct see guy labour masturb smile schock wait around turn back walk away briskli spot light crowd kept follow sometim found head direct mani peopl turn around went way shaken reach home felt fear reflect feel punch good howev run away could think moment
come back home coach class suddenli someon bike held hand increas acceler bike
3 friend guy comment us made indec gestur
harass blow girl suffer anywher harass necessarili attack stranger intens assault within confin hous consid home type stori account shook life left scar way attest irrevers event sunday return home eeri silenc tension air palpabl look eye dad unmistak anger kept quiet walk room hope worri reason slouch bed slice chocol cake tune favorit channel tv two hour later summon dad discuss knew moment fear materi span second come conserv famili girl work corpor career orient independ thought action truli liber yet righteou thought never question approach toward life constantli put famili incessantli compar elder brother thought suppress ideal question night dad furiou sunday friend social caus mix group girl guy unaccept though god know coeduc though god know despit educ man thought shallow hurt demean women rage independ govern societ norm consid independ make mark insol immor want confin four wall hous prepar get marri without even realiz dream sat object ask phone decid cut social circl crush brand new phone piec threw broken piec right face shatter screen leav scratch wherev tini broken glass prick relent hurl abus call name father never associ daughter tear dri listen feel stronger motiv succumb harass worst seen somewher brought peac alreadi face worst constantli defend educ career strong ideal baseless autocrat barbar thought would call barbar could never imagin father would move agitatedli around look knife kitchen harm kill everi emot could associ fatherdaught relationship stood upto look eye told worst thing could ever shall never forgiven relationship end night havent spoken eversinc walk room hous sake mother wept endlessli room abl fathom happen dad extrem behavior silenc mother fear brother eye made realiz shallow societi even call societi one life mean noth girl allow believ women weak inferior absolut logic rational absolut forgiv anyon respect woman parent know mani girl live far dread exist pen stori sympathi appreci stand simpli encourag women men read propag understand beauti divin amaz women god creation want us make societi everi woman safe happi success everi woman equal everi man cri lose hope depriv educ true independ absolut achiev pin creep across citi enlist crime spot around us need realiz women suffer hand famili true movement women safeti begin women stand respect world respect stay strong stay safe
insid bu man came closer tri touch move place
go back home school dda park boy sit start laugh pass comment even made sexual invit
happen girl marri someon outsid commun boy commun tri throw acid
rain walk home peopl talk say look good rain stand shade get terminu
girl stand us stop wait bu made uncomfort two men stare
sexual assault felt uneasi know well safe place prayer
group boy comment friend go home school start discuss us react action use bad word
comment whistl
drunk man forc ladi touch breast
hoot whistl biker pass
sexual harass younger brother harass rape girl get pregnant incid occur famili hous mutengen 7th may 2014
walk cousin road mani men stare small sister around 20 ignor walk along act like never happen
ridg area even group guy teas
sit bench next stranger move man said goodby hope see later hope see soon real soon shout quotim gonna get thatquot made disgust nois
pass ground near indira nagar comment pass ogl
work woman main reason feel safe crowd
physic abus conductor bu travel
boy sent friend request friend accept start send dirti messag
stalk eve teas
incid happen travel goa mumbai bu mom behind driver seat mom window seat sleep around 9pm man tri touch shock dint know say bu halt dinner told bu driver charg male passeng also support us came back dinner shout
chembur station real sister saw ladi burkha hang around use napkin face anoth ladi fell unconsci ladi burqa took unconsci ladi mobil phone woman anoth woman
two men misbehav way back school afternoon
chain snatch foreign market peopl tri help men ran away knife
man start follow refus give phone number
pebbl pelt
ye sexual harass come home colleg man set near tri touch sensit part
live societi nearbi stn premis walk toward stn kept notic man follow almost everyday though seen around
friend bu drove twice turn toward men insid bu pass comment laugh scare
ladi face travel bu
incid took place today outsid pvr prashant vihar around 345pm friend walk toward car swift desir black taint window start follow us even got car still follow friend got take auto still went even taint window allow jow still car window dark black taint
boy pass comment
stand queue husband young daughter famili friend young colleg group push behind person stand behind pinch slap report polic took happen trade fair gujarat pavillion
guy came pull tri rape
7 pm non ac bmtc bu bellandur petrol pump stop sinc lot rush man middl age look innoc start masturb tie plastic around privat hide side bag start pretend scream tie escap
major individu lgbt group work signal rest either tell anyon belong group work prostitut famou suburb metropolitan becom home mani lgbt peopl especi teenag boy harass frequent peopl see lgbt signal either danc front car prostitut someth happen twice happen everi group student differ univers karachi start campaign end harass result progress seen
harass
leav colleg bu person comment touch amp bad facial express take pictur bu 764
touch harass dadar station
street light bit problem away see happen men take advantag small girl play around
comment touch amp bad facial express punjabi bagh amp uttam nagar
go home alon without friend travel toward gwarko peopl bu whistl gave facial express
afternoon 3pm guy right front masturb even ask help
invit come eat hous came start romanc caress ask food behav noth happen got angri left
afternoon way home travel public transport bu person stand next comment
person take pictur girl use cell phone travel metro
 90% |################################################################        |
halt car young guy stand right next window stare car told shopkeep pass tissu paper guy decid grab tissu paper realli tight stare eventu pull hard abus quickli pull window
beach walk man repeatedli came clash
school girl pass men group start call anger sosh start exchang word
walk along coridor certain man nowher approach told admir back long
comment ogl facial express
stand outsid gym alon wait driver pick late time contin chase group boy stare till time driver came
got whitsl comment guy
harass
group teaser near beach
ogl comment
scare way home two boy comment ogl
touch grope due poor street light
teas friend go visit teas us
friend bu guy start sing song stare us continu made us quit uncomfort pretend speak uncl polic
incid took place noon comment pass metro feeder driver
comment grope near metro station daytim
eve teas indec exposur
wit whistl girl okhla mandi afternoon pass
whistl
guy even time make almost immposs walk road
crowd area someon touch privat part turn back could find place crowd
stand buspark gongabu wait friend two girl came standbi side ask sex
train run late platform crowd move upward bridg go platform 1 man start touch warn 34 time shout public ran away
man came closer grab breast could noth moment felt trauma month couldnot travel alon
incid happen friend mine disagr girl thought somewhat solv left went studi back boyfriend met start beat girl embarass made stay hous disgrac
chain snatchingev teas
way show ground mate motorbik men told us get marri old continu school
swim coach touch inapropri learn swim
visitor way take bust stage motobik rider start whistl comment dress
place fill lechour men pass comment
citi unsaf wish citi human peopl friend molest father bangalor
comment catcal take pictur metro rajiv chowk even describ
travel bu stand due nonavail seat bu crowd henc everybodi stand close time guy tri touch inappropri
boy stun girl stand sarangpur bu stop
go school travel public transport much rush space stand sometim found man hold behind know pinch badli gave bad look
incid took place 8th novemb around 815 pm take tution go back home realiz need go market get stuff way local local market lighten nevertheless go daili hesit walk path hear guy call slut sinc road lighten know bike bike light also soon turn around see guy smash bike right leg know intent coulld come sens guy actual ran bike stane lane injur leg
street light what ever cant even see beyond 2 meter night
walk friend guy pass touch
harass
happen hous brother harass also misbehav night six month
night boy wee pass comment
guy continu stare girl pass comment dress even metro
happen metro blue line rajiv chowk around 450 pm metro extrem crowd middl age man kept ogl corner make feel uncomfort
boy slape girl kn road
night raini day mostli case comman
walk underpass footpath man walk toward opposit direct assum seen make way sinc closer wall underpass instead walk right way possibl accident unless sightimpair keep walk without word acknowledg kept walk two second later turn back scream quotwhat fuck think doingquot never turn back kept walk walk sinc dark know els could want pick fight els watch
friend come back parti rape three boy wishtl success ran way
old man stop car ask enter
stand metro line wait metro come suddenli man came backsid start touch stop push backward
guy stalk
2 men taken money jewellri famili go board train
man manag forc tie tree dda park get back school afternoon kiss touch inappropri ran away stood protest final manag untangl get back home terrifi day thought school still get nightmar
friend use work domest help uncl place way work follow comment men inform employ confront men got stop
gang boy abus whistl use bad word way back home
man rub delhi metro
men whistl cat call
guy purpos touch ran away
eve teas dwarka road friend
take pictur
two guy bike handkerchief face drive fast whistl girl
dusshera fest tri felt
metro friend mine guy continu whistl
travel dtc bu 879 shahbad diari someon make weird facial express comment happen afternoon
ogl even
misbehaviour guy
crowd micro bu someon touch inappropri take advantag bu
harass morn
even
stand balconi 900 pm whan guy come closer societi dark nobodi start peel pant horribl experi realis masterb
boy snatch chain
return school old man follow tri touch bodi
old delhi railway station
travel bu man behind staredc bu stop man came touch
teacher tri touch dp dwarka
group boy comment us friend white transpar school shirt
catcal whistl comment netaji subhash place
stare continu
guy put hand shoulder tri touch breast
return colleg head toward maitighar bu stop boy came ask way thapathali show way start walk guy follow ask person question walk faster reach bu stop lot peopl guy walk away
name shanjana mae sam ko jab tiyushan clla jati hu vhan par ladk khde hokar comment kart han
incid took place afternoon touch
school teacher alway punish girl although anyth sexual harass
victim walk inner circl accus use heavi tone make obscen gestur
walk boy stare blink eye
walk guy touch behind happen night
walk seashor friend sudden boy came comment
sexual invitesshow genit touch amp group
even tri touch
done pubic bu indec way without fear public happen even
stalk
buy veget shukra bazaar two men start whistl comment breast happen late even
men sit gather road head karolina kibera alway call girl take pictur disturb irrit
eat refectori refectorian prefect girl stand door light suddenli went form five boy ran dark touch breast scream help bodi help light went thesam thing happen
saw incid ladi pass men told drope coin lie ladi turn men start teli beauti
guy saw near bu stop follow littl ahead board bu later stop saw dad
coupl date eachoth quit long time due religion issu guy throw acid girl fren
yhan par kuch ladk khde hokar ladkiyo ko commen kart hei stake karna
man tri grab breast ran away
stare
matatu karanja stage saw tout friend call ladi sell cloth ignor start make fun
woman know harass around 6 month back night
comment man near colleg
guy pass next made sure rob elbow breast guess effici way guy touch someon girl wonder quotdid realli purpos imagin itquot
man touch butt travel bu
walk suddenli two three men bike comment
come back tot colleg thing happen
boy men comment usual
man grope pass
way shop drug dealer back follow slowli said thatif go hous kill spot sped ran quick
girl pass wear mini skirt whistl group boy start comment dress
walk road even stroll guy bu stop stare sing call name realli embarrass
realli bad
 92% |##################################################################      |
girl famili get harass unknowingli brother law call meet meet behav dread start kiss naughti silli thing make girl feel bad cant rais voic famili prestig
friend two guy came bike person sit back touch friend ass
danger
girl almost rape hous thief came neighbour steal girl save came take school shoe intend report polic
happen even
girl live coloni stalk 34 guy almost everyday coloni signific popul ia offic ip offic jurist etc
friend mine femal rape dirti south buea 10pm night went transfer credit return held guy took uncomplet build rape
metro station bad gestur ignor
misbehav
man tri touch privat part escap
realli bad
matatu go town conductor ask money stretch give touch mhi breast
conductor tri come close travel nashik mumbai
touch grope
happen travel bu colleg morn drunk man got bu two femal friend accompan man got stairng us cruel way report conductor rather make get shift place amp person sat right behind us atlast got bu option left amp passeng travel along
went market two boy start comment
call name teas
harass
follow metro station coloni
femal friend night group boy start comment
grandmonth met two men stop us said want grandmoht give refus start insult
boy offer ride metro station abus
18th august 2014 friend sexual harrass landlord hous landlord son everybodi gone farm left behind went watch tv neighbour hous felt asleep suddenli felt strong forc top landlord son rape
finger polic abus
miss street light mani part
comment ogl night
ogl stare
happen quater boy quater two month back dirti south well girl beauti pass mani boy lust boy tri approach snob boy threaten
man expos privat organ widow woman
girl tri push cab
guy come bike ask woman way toward certain school make invit drop home snatch chain rode away happen morn
happen road approach toward metro walk afternoon
two girl get harrass comment 3 boy
went colleg fest friend wear dress boy pass mischievi comment friend
go market even boy stand roadsid start comment
guy take pictur girl comment unnecessarili happen night
touch grope
someon click photo variou women malad station amp also molest small girl
pass road bike rider slow bike grope breast complain polic freed sometim
morn salesperson home tri molest touch etc
happen auto rickshaw
walk pass street friend suddenli heard whistl corner turn look saw group boy throw slant friend somehow emabrass
sent shop night reach home man caught hand tri kiss lip
group boy sit teas stare girl even catcal comment touch happen everyday lot girl
20yearold man allegedli tri abduct molest 12yearold girl safdarjung enclav return school wednesday afternoon
two men bike
poor street light north east report face mani harass frequent
sexual molest man street live dark around save
friend mine sexual abus still primari school primari school understand sexual harass notic friend develop breast time constantli held boy found push ground touch breast
stalk guy near park shiv vihar happen late even dark decid confront quickli get home
wanna bring incid ur notic happen n frnd stay chembur mumbai went even walk rcf sport club walk road parallel sport club rcf coloni n youth council hall side narrow road road crowd n bit dark although street light guy came bike hit frnd back walk outer side speed bike couldnt note enitr bike number bt manag note mh o3 3347 could help highlight concern peopl thank
guy comment bu stop
afternoon hour subject comment catcal whistl
made touch organ forc bad even resist stop
guy comment friend shop mall friend slap went away
bu toward pragati maidan
man came touch breast drink water immedi walk could see face
incid took place 2009 oustid dav public school pushpanjali enclav group boy use chase way back home car use pass comment whistl ask friendship continu 1015 day told everyth elder brother handl situat
chain steal
happen near madangir bhumia dev centr market man use pass comment use take pictur everyday use constantli tell love use everyday way offic use tell boy children around love use cut hand blade also threaten would see someon els would attack person blade around 3 month
even guy follow tri get intim follow hous much noth bad happen
uncl tri sleep aunti left shop
beggar harass touch
long time ago school girl return class use road short cut reach hous road hospit locat mechan shop would think would safe see man lungi hold peni masturb amp head toward panick ran sinc didnt know els stop use road brief period thought onetim experi howev mistaken incid happen stop use road fear safeti horribl incid never anticip happen twice row road constantli use peopl sight like scar child mine feel good abl share experi women ensur doesnt happen other
group boy comment whistl perform street play campu
group three guy move suddenli start whistl pretend hear suddenli turn saw turn hid face
incid eve teas netaji subhash palac
saw man stalk got train toward cst kurla saw man kurla station got dadar station follow also went grant road station realli scare see board taxi made mind decid see get taxi report polic luckili follow till experi scari know whether follow want go place
guy whistl start follow market
happen night feb 2013 two biker snatch chain woman walk famili
cousin harass place
boy sit around big street light play ludo alway disturb us way school back call us name likesweeti love cuti make us uncomfort
girl walk two boy pass comment abus boy walk street
chain snatch molest
morn walk girlfriend three boy bike pass start comment us hoot bike gone
bu boy touch thigh
person public bu tri touch time time
two girl pass boy group start comment cloth bodi happen even
got doctor clinic kalkaji began walk home tall well built fair kashmiri man ask knew restaur shook head said kept walk mutter someth breath made uneasi turn around walk back doctor clinic follow back saw stand behind glass door came back fortun car came pick could see look 1150 busi street friend came pick also aghast way man liter hunt wear full sleev kurta
area around station platform fill men boy stare women even pass lewd comment
touch grope comment
heard man masturb next place later came toward patio scare lock door
misbehavior guy
masab tank main road
25th decemb went around 900 pm friend chill way stop told guy alway tell girl see love stuff pretend love told refus start ask think refus propos felt harass
friend come visit hous boy near hous look indec pass comment among
sister stand front shopper stop mall wait car come two guy show car open back door car invit us take ride start comment happen afternoon around 34pm
night sleep
unsaf pedestrian bridg connect borivali east west presenc drug addict bridg presenc anti social element bridg desert non peak hour like afternoon 10 pm light function find broken alcohol bottl cigarett condom lie earli morn drug addict found dead 23rd septemb 2013 rat eaten away bodi unconsci also lot men stare pass comment femal unaccompani bridg desert
custom touch bodi roll finger palm look eye reject touch breast even refus buy compli
travel train person touch inappropri manner slap
love comment
catcal comment take pictur
go colleg hostel two person start comment
2 men sing obscen song look
driver auto tri touch spot got
someon whistl also stare
incid took place even metro station girl eve teas pictur taken
travel bu group boy came near tri touch privat part suddenli told intent peopl travel bu help
middl age man utter vulgur woird
chain snatch today morn 735 two peopl motorbik
comment stare
walk stage man grab arm felt uncomfort
harass public bu
comment ogl
comment cat call whistl pictur also taken
pass street light suddenli man came press chest behind ran away laugh could understand anyth
 93% |###################################################################     |
dombivili famili consist middl class categori famili peopl travel toward mumbai main citi livelihood henc daytim especi afternoon women elderli children home henc make home peopl vulner increas number theft chainsnatch instanc afternoon women bring children home school thiev becom brazen keep watch societi snatch chain within societi compound
harass
chain snatch karol bagh
near jewel shop even
friend chain snatch street
catcal ogl gtb metro station even
stalk
defil someon know well told go hous pick cloth chair came start touch breast tele give anyth want
molest woman bike husband group car
guy stare made vulgar facial express
friend boyfriend alway black mail tri physic relat
ye sexual harras mani time life like random guy tri touch sens part group guy whistl diffrent facial express
gener coach metro toward jahangir puri person touch back side
3 guy bike start howl start sexual invit busi street
chain snatch two peopl bike wrong side afternoon around 34pm
return back home group boy teas follow call friend help
woman bodi found lie damp site
friend walk behind coloni man came friend grope waist
school go boy watch ponograph content mobil phone togeth younger brother who still primari school
kiss inappropri famili friend
18 20 year age femal student pcl nurs attack rape driver conductor micro return even duti
incid took place pitampura villag even two ladi travel rickshaw suddenli two young boy snatch chain ladi
realli bad continu insid campu
micro bu somebodi touch back hair
teacher beat student behind sanitari towel fell floor
travel public vehicl friend star passeng next us sometim start comment us start feel uncomfort make environ insid vehicl tens took vehicl
secondari school classmat rape age 8 alon home 32 year old neighbour man 32 year rape till date suffer effect psycholog sometim behav like mad woman
way home old man whistl
three boy street teas friend say quotbeautifulquot
guy touch
catcal
incid took place even guy take pictur market
man saw friend underpass masturb front use foul word address
touch inappropri
got metro guy also got stare follow till board bu
ground look dirti fashion also spoken dirti way
mostli even sir use call odd time
happen friend come train train huy misbehav offens way inappropri
shock incid report pink citi eve teas common happen girl almost everi well let share 1 incid friend studi 1 prestigi girl colleg jaipur stay pay guest cscheme jaipur two friend went market get someth suddenli 1 car came think 45 boy car tri drag friend car grope wrong place luckili friend save shout dread still cannt forget incid 1 case kind incid keep happen
go school 23 boy manag faculti comment
indrajatra basantapur watch tradit danc old man touch arm tri touch breast abl shout notic properli
group men push walk even tri touch privat part pass bad comment
school boy smoke bang next fenc see girl run
whistl comment aquaint
walk near lake area colleg
head home stranger spank pass done anyth wrong felt bad couldnt anyth
touch public transport boy
experienc sexual harass travel public vehicl usual includ men use foul word tri make contact
famou mall kalyan call metro juntion section mall food junction area usual fill lousi boy make uncomfort visit
forc sex cousin pali insert hand vagina age 6 remov
happen even
return home saw man touch women seem quit uncomfort abl speak
group boy comment whistl perform play street play
ws marri daughter uncl rape put lotion vagina destroy evid took nairobi women hospit reveal rape
sub center allot ashok vihar faculti misbehav made comment charact transfer made complaint
survey carri safec red dot foundat along safeti audit street market mumbai
return colleg home boy play privat part show scare run told incid friend friend also told guy alway place
go school morn riksha puller start follow slowli look leer ran way school could hear laugh
guy start whistl comment went bu stop
incid took place near jama masjid old delhi even girl famili teas group boy
bu pack man came close sister touch
ogl facial express comment
watchman societi keep give mani girl glare stare inappropri
touch grope nearth metro station broad light
boy came bike hit back ran away
class fyjc even borivali station middl age man stalk friend man afternoon twice
15 year old boy go way nepaltar someon follow scare
conduct survey part sociolog assign cours offer iit women worker construct site said sexual assault
guy pass bad comment way home
girl 8th standard stay grandmoth refus go school fruit seller aquem opp mongini keep star way back home scare prefer answer exam
guy comment take pictur near metro station
wait bu bu stand man 3035 age stop bike ask life bad intent facial express
tea stall outsid hose whenev go tea maker ogl
month ago seen peopl wear go bu saw guy tri touch young girl n scole guy n left bu foran
girl walk road alon follow group 5 boy
happen friend mine tigri come back school four boy surround harass also shout help interven boy took away parent lodg fir kill boy
saw boy comment pass lewd comment rohini metro station afternoon
catcal model town
eve teas comment
drunk man comment tri take pictur station
two men bike snatch gold chain market realis gone
sit restaur middletown friend group boy start take pictur
anoth incid 1st jan 2013 malad west near liberti garden new new year less month previous horribl delhi rape case happen walk street buy someth local shop realli old man look made disturb gestur shook head side side meant would like put face breast said hell smirk first year
walk road much light motorcycl pull besid two young men told look great ask want film film left said return twice ask chang mind
cousin brother use share problem regard famili well room talk colleg studi behav like care help start open cloth could anyth start sex sever time share stori anyon till today
travel crowd metro busi cell phone suddenli felt someth zip trouser realiz elderli man tri unzip move immedi coudnt say anyth embarrass
harass bangur nagar
around 30 day back sister went work afternoon 2 boy start sing obscen song
catcal comment buse afternoon
forc touch outsid school
man start masturb see friend alon isol area even tri follow
happen friend return picnic boy tri harres
harrow experi
whenev go outsid hostel face kind harass everywher everi time happen everywher outsid colleg hostel
incid chain snatch wit travel even report incid polic act wit ladi
sexual harass cousin made advanc tri touch breast privat resist threaten shout witdrew
whenev go toilet complex harass
harass man walk ask still attend church use attend could recognis person ask name church could answer said rememb
group boy teas sing whistl walk street
go home funer around 2 00 way met five boy held hand tightli cover mouth took empti hous three rape left
age 16 yearstyp harassmentcom school leh time
happen ina metro station afternoon guy pass inappropri comment girl
place extrem unsaf foreign tourist wit peopl take pictur call name behav inappropri
comment
go back friend place two wheeler broke near park road tri fix 5 minut saw two bike come distanc 3 men bike 2 stop nearbi start whistl comment fortun problem got fix quickli got scooti left fast quickli possibl
friend way home notic goon villag initi whistl goon surround grab hand manag free fled spot rural background afraid terrifi incid file complaint polic
mob thug drugdeal stalk women catcal bypass polic noth polic ever read guy call mehdi tall scar forehead friend drug dealer
comput teacher often touch privat part
harass
friend car guy came knock door blackmail us tri harra us sexual verbal
seen man show privat organ ladi ladi afraid insid bu
guy teach us comput touch girl chest inappropri manner seem disturb us
 95% |####################################################################    |
know coupl gone get marri guy stab templ knife ran away know case
sir board school tri assault
happen even
boy basantapur jatra tri touch time time
silent street saw girl taken cab harrass
girl go colleg mani guy road start take pictur teas comment rough word afraid go colleg kind abus peopl
comment
happen morn
work field coordin around year back man follow ask contact detail
middl age man teas inappropri
group boy whistl said differ thing friend felt embarrass
man ran away purs riksha red light
man follow station home whistl whisper ear quotcharmingquot leav
friend place friend father tri touch inappropri
happen afternoon even
group guy comment
touch grope afternoon
night gulbarga foreign went visit shoe shop owner shop got obsess take even though told want go insist pick hotel close time around 2230 told go late left hope would find stay told hotel pleas tell anyon live hotel someon ask anyway 2230 call recept told wait told wait anyon tri call half hour constantli also came door tri get luckili stop morn scari men hotel complain open door shoe shop owner wait repli ask reveal locat anyon could understand done anyth wrong morn also call want massag
3 month back harass man extent depress chang lifestyl complet scare everyth even go work way back work guy would follow bhoomiya dev templ main road wherev went even click pictur phone told peopl around love would slit wrist dont come even threaten harm blade saw someon els know end worst thing happen
saw old man tri comment poorli dress ladi
girl stalk grope bunch guy went run look help ask mid age ladi help see boy fled ladi drop girl home safe happen around 2 half year ago exact date specifi
grope train
someon follow
inappropri touch
friend mine follow person
friend go toward colleg park right opposit colleg boy stand corner start comment us use bad languag
travel bu nairobi town kibera man sat next kept show ponograph movi
comment
night iwa pass moti bagh founfd area actual highli unsaf
incid took place even saw group guy comment girl featur bodi
horribl experi
touch grope
way met ex boyfriend forc get back refus remov cloth start touch breast
stalk
pedicab cyclist yell insult pedestrian
guy pass comment whislt
boy bike constantli take round pass comment
return colleg boy stalk
happen pvr anupam complex saket friend place group boy stand start pass lewd comment
friend mother vikaspuri f district park saw lewd comment pass boy stand tri ignor went away alon could give back
men cycl masturb drive close women girl st andrew colleg turner road
morn hour around 9am unnecessari exposur guy filth
realli bad
go town found group men sit iddl pass start whistl comment dress annoy
went visit person hous know drunk start behav funni notic took bag leav want forc fortun alreadi near door quickli push door ran away
teacher beat student back waist
guy bike whistl
chain snatch night cross road bombolulu
boy form group teas girl
walk along quiet road friend motorbik stop besid us man jump back motorbik grab breast hand start push breast toward side road aggress luckili grab bag ran back motorbik sure would rape reliev man sped away motorbik
come back tuition dg2 vikaspuri guy expos privat part harass girl
class three boy sodomis four upper primari boy
went heritag walk group 25 girl comment
pass comment
walk street sector 1 27th main road stalk cat call bunch middl age men street follow till hous realli scare pleas help
friend play guard threw chit phone number ran away harass
peopl electr rickshaw constantli make facial express tri grap attent incid took place afternoon
student karachi univers protest yet anoth sexual harass case time student master urdu depart protest teacher known sexual harass student teacher well fourth year row student accus teacher sexual harass karachi univers teacher current serv high dean faculti list teacher student gone complet boycott class administr remov said teacher depart said teacher famou name countri literari circl known verbal physic harass student student advisor ansar rizvi promis action said professor work varsiti extens retir three year touch around 80 year age report eight femal student surfac far student take chanc teacher demand immedi remov faculti karachi univers formul committe investig matter howev student vari result sinc teacher accus harass clear inquiri continu teach respect faculti dean faculti administr scienc look matter promis swift result protest student
happen friend mine rape friend went visit claim sick ask come see got close door kept key cooper beat rape
went shop sarojninagar guy stare
stand friend polic bu came polic man last seat pass filthi express
bu stop man tri harrass
go baghbazar samkhusi public bu man sit tri touch bodi pretend take purs
walk cover head cap drizzl man came ran away cap shout help said meant men ladi
sexual invit acquaint
motnth back tell prcise go school ladi approach want direct hospit converstaion fareman thief heard us told go school take ladi hospit know well thief noth could help ladi becam guilti condit faremen danger tri intervain help
senior prof ncra seem misbehav past ladi
went nativ place father beat much mother brother call polic save
boy alway follow go home school
guy behind came ask sexual favor
poor street light near rajapuri bu stop
ogl saket
bu crow last seat bu boy sat lap start touch everywher even tri follow
happen afternoon kamla market uncomfort peopl around oblivi
harass street friend went stroll mutengen excit greet friend shool fund fight friend told gang laod money attack us street knive happen night guy collect money us went away
group boy comment walk road
happen walk around never area beauti chain around neck suddenli two men came around order silenc took away chain touch boob
touch grope crowd place train buse take pictur
incid took place septemb around 4305pm guy follow till bu stop kept mumbl cheap thing got scare ran took auto metro station
someon grab ass walk
misbehav
share auto bandra west stare face man auto
saw girl face uncomfort situt form catcal whistl
stare badli comment took place late even
stay brother inlaw cooleg wife sister stay rural home keep touch bodi wonder doe mean man like
harass morn
proper light night secur near colleg lead eve teaser around colleg
way home work guy blow whistl
harass
girl grab man around corner fought back stab
incid took place august around 915 metro gener compart jam pack man tri lift top
man tri grope train station
meat shop baluwatar salesboy employe meat shop whistl teas us walk road feel unsaf
old man stop car ask lift said come drop
got friend trust much highris somom class boy never report anyon first time
incid took place rajiv chowk metro station afternoon guy behind pinch ass gave shitti smile
confront fought 3 young men want rape young girl nairobi intern trade fair forc touch privat part
morn hour guy tri touch girl pass vari cheap comment
use relationship ex use go date differ place start ask physic relationship later start forc could share parent ask friend suggest ask quit relationship
happen 30th march 2011 around 5pm guy car chase friend stop car front car start comment though escap polic help
 96% |#####################################################################   |
cross bu stop chain got snatch men could anyth
olymp stage wait bu men idl start call tell come sah hi
go home school two boy tri rape
ogl facial express comment
guy take pictur scold happen afternoon
puja guy approach mom friend start talk suddenli snatch chain could react left place
boy privat bu comment girl
walk toward hous old man drive car stop right side look ask get car
friend two guy came bike person pervert sit back touch friend butt
sister pass alley boy whistl stare comment dress way walk
touch grope
stalk drunkard scari ran crowd place make safe
person sit ladi seat ask seat start misbehavior
school friend hatigauda area boy took photo
chain snatch
man shout car friend walk
return home friend saw 4 boy harass girl touch unlik show nude pic mobil
victim follow group boy also made indec facial express pass obscen comment
area realli unsaf night
friend attend even school way back home sexual harass bike man seriou could go home incid found morn came harvest tomato farm found lie unshak
happen month ago littl girl wit harass boy littl girl pass guy call went call boy want kiss refus accept boy slap lay low gave slap well
friend walk saw two boy follow girl later saw men hug girl backsid shout loudli ran fast inform polici nearbi
gurgaon sohna road unsaf night especi alon street light less peopl around vulner
harass
peopl found comment women late night near raksham nilayam
stalk man cs metro station follow till new delhi metro station soon sat airport metro man came sat seat right next mine sens someth wrong rush last compart metro
friend told follow guy everyday care initi take action gut stand
two girl two boy touch other privat part
girl touch knowingli man crowd metro around 6 pm
chain snatchingafternoon 4 pm
girl 6 year got rape father
friend taken bar men morn told rape
gang boy pass comment girl pass near commun centr anushakti nagar
8309 pm onward street full rickshaw driver either drunk high local hang group road pass comment unsaf walk road night
man touch travel local bu
group boy way whistl us make unsaf uneasi
come back colleg group men motorcycl teas
stare privat part time
grope market
harass
took place beach group young men
near mhada boy caught hand
saw boy whistl girl pass school
cross road motor biker show dirti facial express like give fli kiss
walk street girl teas threw small pebbl get attent toward
mother sent shop night way met old man tri kiss succeed
comment hoot
eveteas street
man follow walk road
7 year old music cla man call upto top floor tri abus fled place
touch hoot
return back home bu saw girl touch man quit long time girl shout back everyon bu support
last week mysor zoo 3 guy take photo ladi mobil phone notic question rush escap
man would stalk way home whenev look would stare back happen near ravi bazaar pradhan chowk
waz shahdora border boy waz teas girl bike
drug addict problem whole mahim west area peopl safe night mahim west area drug addict steal petrol car batteri street furnitur etc mahim west area approxim 100 drug addict sit mahim sea shore smoke chara road mahim west unsaf mori rd ladi jamshedji rd cadel rd chara smoken pl someth
victim mind way accus came scooti behind touch intent
girlchild live stepmoth stepmom son forc girl sex almost everyday without parent concent
guy stare pass comment
comment catcal afternoon
boy comment girl whistl
mani boy comment walk street
go coach class peopl start whistl even hour
good friend worth trust friend mine tri touch kiss whike travel auto
almost happen everyday
way home two girl also boy came stand near girl start stare
stare
catcal amp whistl take pictur
walk kisumu ndogo men usual idl besid railway line string made uncomfort
parti friend birthday took alcohol intend sex follow morn found nake sex bleed lost virgin
man known disturb girl sent market mother around 1pm met man way forc stand refus touch breast
peopl speak bad word public suitabl societi
secretli take pic
frequent travel mumbai local train like mumbai women take harbour line train goe mahim cst seen mani time 8 pm train get empti presenc rpf personnel railway look increas secur rout
use shop front hous use play shop owner son shopkeep tri touch privat part intent felt nervou still today meet person uncomfort though shop move
5 year local bu man touch organ sever time
chowpatti unknown boy came start talk comment
harass even touch grope
boy show facial express western dress girl
someon tri get physic main intent distract steal purs
saw man misbehav girl budh bazaar could help sure happen pull arm happen afternoon
guy comment tri assault govern school girl
steal
ogl man 15 minut straught know get stop
pictur along friend taken parti night unknown guy
morn saw accid result death 2 peopl bike fault bu driver
man urin face road near hansraj colleg flash pedestrian
group friend festiv season night time pass drunk peopl fine till later male friend go back way return beaten drunkard
place surrond goon unsaf place travel
even certain start follow time stop also stope start walk also walk scare chang root make sure follow
incid took place night disk guy behav inappropri
realli bad
happen even
got call unknown person say know
even two men bike snatch mother chain ran away
sexual harass wit friend muyeng way market fate surprisingli unknown person kiss breast becam angri
misbehav
coffe meet peopl tabl pass comment make difficult concentr continu stare
group boy stand outsid colleg ogl us comment cat call
guy unbutton pant tri show privat part tpo group girl
school health teacher use touch student inappropri way
man saw pass go ny toilet near build decid urin wall toilet assum pass
stuck crowd thursday market low cast cheap touch inappropri 3 time breast ass abl find first crowd found shout nobodi came help pretend heard noth sadli ran away could slap
everywher nowaday happen everi time happen everywher sector 9 also
girl pass road age men whistl
travel bu boy whistl comment
attend parti shock arriv late find group guy take turn sex drunk girl friend
way shope buy mandazi break time mate drunkad man start call hey babi youv got nice chick dind respond elderli man walk away instead
ladi fight amp abus ladi coach
men hang around constantli pass comment ogl women
kalyan small crowd citi ton rick stand outsid kalyan irrit part stand rick wala time rude deal
two girl whistl pass comment bar
comment
templ boy strted wink show abus facial express
travel banara rae bareili railway station man tri sexual assault
 98% |######################################################################  |
girl pass guy whistl comment upon happen night
chain snatch
happen walk alon boy teas vulgar word
touch
mani incid grope public transport
way home colleg seclud area
travel metro man took full advantag grope touch privat part
2 school boy pull hair outsid school said dirti thing
even travel metro disgust touch person even back sour
comment
2 school girl move way group young boy walk opposit side girl way among boy tri pull skirt
guy dtc buse take pictur ladi bu
went pragati maidan trade fair guy touch grope surf stall though incid occur frequent almost anywher delhi
friend walk street group boy start whistl make inappropri comment
walk new place alon follow stranger helpless afraid got idea pretend talk phone know
harass morn
group boy wistl said rubbish word
late even go nearbi market alon buy provit strang middl age man lean short come toward saw pass know turn back saw look came behind wait shop enter becam nervou call boyfriend sinc much nearer dad happen time till fake take pictur shout pictur show polic right front mani passerbi sinc seen
go bu mostli men comment girl
continu comment friend even 6pm
chain snatch riksha return work
happen afternoon move bu
buy pair shoe man stand next stare continu head toe whistl cheap song
school senior abus badli told get lost class bad thing girl
two guy shout love u bicycl pick brick threw
went hous night walk saw nude man masturb front hous
walk toi market peopl sell cloth put arm around shoulder tri remov couldnt got argument finnali releas
watchman liberti build tiwari seem child abus feel children forcibl kiss choos remain anonym observ
incid took place near bypass 3pm visarjan friend auto guy shout us truck
ogl touch grope
come hostel two friend bunch guy start comment use abus languag happen around 78 pm
rush metro wass friend guy touch back felt uncomfort scold
report taken dadar market exact locat bigg salesocieti store interview feel safe due rush
return matern uncl hous alon three boy walk toward laugh us even comment
safeti matter
visit matern uncl hous group boy near swayambhu teas catcal whistl
walk someon tri touch bad manner
guy swear time use bad word address girl mani time happen time
hot sunday afternoon taxi go market muea man sit infront besid driver driver drop everybodi taxi sudden start touch tell fresh leg ask number give
travel bu colleg home pack guy stand back trap hand felt uncomfort ask remov hand
physic abus sexual abus revathi famili member motherinlaw foce suicid give kid revathi pleas save life revathi kid
along friend visit swoyambhunath boy teas us whistl
boy show privat part
friend cross bridg railway station group guy start whistl pass comment us
feel safe station light crowd
car 4 boy cross realli close hit
cab driver start talk previou custom boy forc got girl rape speak hindi detail said penetr ejacul know new got cab cuff parad colaba churchgat
group boy ask sexual favor go jungl area defec keep disturb parent tell ignor
man step feet even say sorri smile funnili assum
walk get vehicl park seawood stn park lot men joke around made comment regard dress sens felt disgust want thrash could gather courag
stalk comment local boy
misbehav
child rape uncl ran away 13 year becam pregnant
pedicab cyclist yell comment pedestrian physic appear
happen went friend visit boyfriend friend us went outing friend boyfriend start make advanc refus sexual harass night thank friend interven like would rape
car tint window number plate use follow daili way tuition
get church compound middl age man pass ask vagina dirti want god clean
stop use block e main road men even young boy stand stare sometim even pass dirti remark
man show peni call sister
someon pinch breast took place somewher near nsp
eve teas chain snatch
continu stare stand near metro station incid took place afternoon
group boy ass lewd comment come back home offic
grown man car invit night stay offer money abus refus
friend boyfriend pretext take enjoy rape come school anymor
two guy comment two girl road
happen metro even way home men enter femal coach forc
boy follow till colleg station
mandir ke samn bhot se ladk khde rhete hae picha kart hae
10th return school neigbour like uncl held hand near coloni room start tell love use kind word realli uncomfort
victim got car wait two friend outsid block c connaught place friend arriv walk toward park area two peopl came bike snatch bag
night everytim happen highli unsaf night
wascross othersid road use underground subway delhi metrowhen th man tryto crosspast meon staircas know delhi male alway keep close rail atleast side time ther four inch gap te man tri brush past beforehat turn around caught redhand drag polic control room istead immedi send foe ladi policeand book man offenc tri look proof camera turn tat suveil cameraat particular staircas order last tree month man start threaten file fir manhandl polic look shmelesli argu even cajol let man go condit say sorri wast hour wait te poic call ladi senior offic charg offic kept give excus also warn hour would wastedif stay file complaint worth ti small matter madamquotquot kept say even ask ident occup e c even maintain anonym accus mmidiat record phone guess unknown no call time time sinc could man man felt like crush polic men powder peopl lie charg secur safe
mani boy class form five usual like touch buttock want even touch breast like comment other make catcal whistl rest
get scare take skywalk crowd peopl avoid climb stair drug addict drunk seen linger sky walk
man start sing obscen bhojpuri song look direct
two guy came look form three girl neighbourhood saw mother instead easter came look ran away
way home school local boy teas
harass
catcal man
walk road man call low tone assum
comment stranger even touch unnecessarili
even 2 boy commentig girl
went visit friend found brother told sit came remov belt want rape succeed
grope
ex boyfriend start stalk black mail everywher went back scare leav hous month go anywher happen june 2014 later sister friend help get situat locat colleg home tuition everywher went
wait friend near uttam nagar metro station even group boy start pass comment
name
classroom boy secretli click pictur girl ask refus start argu girl bad languag
touch privat place crowd area happen jagannath puri insid templ
ogl comment gk ii
happen even
guy tri snatch purs
eve teas seen report
return coach
even call teacher tution
incid took place friend around 6 month back friend even way mother dairi guy start pass comment us initi ignor day held friend hand ask come agit got hold friend learn self defens colleg help us went back home share parent
go home market reach stage boy start whistl assum
crowd area saw girl touch guy sensit part
iam touch cousin brother feel harass lure differ kind thing like behav like care touch sexual organ
brother law misbehav street near hous ask sexual favor told believ happen near balaji chowk
whistl lane commun chang rout
embarrass horribl moment found teenag boy take vulgar pic cinema hall
guy follow throughout station almost cross road showroom start come way
even time return home frm market guy follow scare n run away frm thre
teacher beat student buttock
comment travel afternoon
happen morn move dtc bu
man bike approach made sexual invit kept ignor went away
go metro station men start whistl us
harass even
monday even saw girl walk four boy take pictur
100% |########################################################################|
bu heard men make comment look behind see
teacher beat student head
incid took place 25th septemb 5pm west gate mall park area group boy stand car park area comment whistl everi girl pass side
misbehav
cat call wink
way quiet lone peopl even bu stop taxi driver stand peopl walk even 8 road
tri sexual harass waitung platform friend
man use pretend urin behind flat man could take privat part whenev saw ladyin cook area behind flat often peopl complain landlord came fenc round flat
misbehav
group boy momo shop kept stare butt laugh
friend pass street group boy comment us say look beauti girl come
happen near virat cinema go back home men stand corner pass comment touch inappropri ran away start follow pretend like aunti walk nearbi rel call bhabhi scare men went away later told aunti happen 16
group guy drink near ill lit area stare contin
bali vaccat famili came hotel prior famili want click pictur car driver say quot love fuck indiansquot furiou went back hotel cheap mental ruin fun trip
happen night decemb 1st went along friend dinner finish dinner left restaur along friend live town girl roam around 8pm though main centr town shop open could find auto soon reach colleg hostel stand restaur road suddenli bike pass guy sit back hit butt happen fast could react time came sens rode fast gang boy front wait us pass laugh incid though seem look educ know educ matter much upbring come town bhimavaram last year studi dentistri mani educ peopl also mani richi rich make town better peopl make best peopl dirti mind bit okay understand call quotmassquot peopl maintain attitud mean highli educ peopl thing make sens
comment indec exposur guy
walk go school group men sit around garag start call laugh talk
school walk back home mid 20 guy seem educ still someth soo cheap make bad comment make afraid travel alon
comment
guy teas girl feel uncomfort
instanc touch random guy afternoon hour
stranger man follow dombivli station gpute road realiz follow turn around scold back deni
way back home saw girl pull push boy toward car girl retali slap hit much hit abus girl hit blow thrown road heard boy shoutingn quotnow know happen refus come along housequot
even road less light old man follow till gate univers indes
facial express apwer made filthi exposur even
public transport inappropri touch amp group rush
reckless drive
jamalpoor bajar cakla
even night time ca tuition everyday use face highli unsaf night
colleagu harass manag complain hr refus believ colleagu believ manag contribut revenu profil compani say anyth ultim resign
grope dadar station
afternoon travel public transport indec grope
ladi wear jewelleri went friend home attend puja return 2 boy came bicycl snatch jewelleri
follow 3 men main road
night 2 boy came bike snatch chain
left select citi mall peopl indec comment way shop gate exit gate happen afternoon
peopl report mg road unsafest place gurgaon need bit polic constraint like cctv camera make safe
indec comment pass right outsid home
around decemb 2013 afternoon somebodi grab bum behind could figur turn around metro station crowd
happen morn
travel pack bu man stood near touch 23 time thought must unknowingli later realiz purpos pinch next time tri touch said sorri
incid took place around marchapril 2013 rajiv chowk metro station afternoon board gener compart dwarka metro rajeev chowk man age 40 comment maal
guy follow
peopl comment dress postur
cousin follow stranger shop bandra
sister follow 2 men
maid servant physic abus son home parent take action
marin drive famou tourist spot mumbai peopl tend take mani pic cousin chill group boy take pic along tri take pic also disguist
go home last friday forc kiss person dont know
incid pictur taken afternoon hour
drug addict flash
student travel daili
ladi go templ teas twothre boy younger comment n
guy tri push shoulder place crowd slap
group guy constantli stare board metro happen afternoon
man follow quit sometim inform peopl
sexual harras bunch men get bu
return home friend way teas girl support
girl go home men behind start follow call ignor went
stare comment
colleg friend return back home bu middl age man teas whistl felt nervou went hurriedli
even ear got snatch
ogl facial express
six month earlier sister went main street get materi way back home boy whistl teas us scold
misbehav
pass train
drag stranger hold hard forc danc along offic anniversari parti
market poor light boy start comment touch
coupl went kakribihar surkhet weekend guy comment take pictur behaviour creat fight coupl group guy kill boy rap girl
wallet phone stolen bu also complaint sudden action
travel chembur station kurla station take train go dadar middl ladi compart soon train halt kurla start walk toward platform sinc train come foot bridg realli crowd man walk behind touch moment realiz touch turn back scream shout hindi ki haath kyu lagaya embarrass nobodi bother interven think polic protect given kurla station alway crowd rush central harbor line
harras even
touch grope comment drunk men near wine beer shop
9 year old sexual harass salesman could say anyth build search could find
harass morn
elbow
way school guy late teen whistl alon
friend bjmc got internship newschannel produc show would tell person sexual experi kind exploit youngster
friend guy came touch breast went
friend given sexual invit well touch improperli rel first anyth incid took place warn rel do
friend follow 3 boy home colleg happen twice
wit incid girl teas guy afternoon hour
woman pass men start stare talk manner like suggest admir woman ignor
head toward home way home go subway notic guy follow follow 15 minut call brother
debat 78 guy drunk stand outsid colleg gate tri come comment whistl ogl happen even
person near construct site probabl labor make kiss sound pass report time date 03 29 2013 1139
threaten make inappropri convers threaten happen afternoon
happen morn univers metro station guy comment teas
aunt returniec frm offic shaunt e found gal fallow guy frm mani day scra n aunt help n sout guy
victim sxual assault rape
(7201, 2)
df1.head()
txt	y
0	walk along crowd street hold mum hand elderli ...	1
1	incid took place even metro two guy start stare	1
2	wait bu man came bike offer liftvto young girl	1
3	incid happen insid train	0
4	wit incid chain brutal snatch elderli ladi inc...	0
Spliting a Dataset in Train , Eval ,Test
def fn_tr_ts_split_clf(df_Xy_, ts_size = 0.2, rand_state = 63):

    df_Xy = df_Xy_
    df_X, y = df_Xy.iloc[:, :-1], df_Xy.iloc[:, -1].values
    sss = SSS(n_splits=1, test_size=ts_size, random_state=rand_state).split(df_X, y)
    tr_idxs, ts_idxs = list(sss)[0]

    return tr_idxs, ts_idxs
def fn_tr_eval_ts_split_clf(df_Xy_, eval_size = 0.2, ts_size = 0.2):

    idxs_tr, idxs_ts_ = fn_tr_ts_split_clf(df_Xy_, ts_size = ts_size + eval_size)

    df_tr = df_Xy_.iloc[idxs_tr]
    df_ts_ = df_Xy_.iloc[idxs_ts_]

    idxs_eval, idxs_ts = fn_tr_ts_split_clf(df_ts_, ts_size = ts_size/(ts_size + eval_size))

    df_eval = df_ts_.iloc[idxs_eval]
    df_ts = df_ts_.iloc[idxs_ts]

    return df_tr, df_eval, df_ts
df_tr_, df_eval_, df_ts_ = fn_tr_eval_ts_split_clf(df1, eval_size = 0.2, ts_size = 0.2)
df_tr_.shape, df_eval_.shape, df_ts_.shape
((4320, 2), (1440, 2), (1441, 2))
df_tr_.sample(5)
txt	y
4405	train delhi rewaricom	1
5876	bike rickshaw go back home look comment badli	1
813	person touch sexual organ public vehicl	1
6827	go town found group men sit iddl pass start wh...	0
1567	guy comment stare girl	0
df_eval_.sample(5)
txt	y
7134	even road less light old man follow till gate ...	1
5224	pass boy start say like tight dress	1
4001	follow 15 year old boy skywalk vile parl	0
5216	guy resid near charni road night incid wait fr...	0
2059	guy bu purpos stick privat part girl common bu...	1
df_ts_.sample(5)
txt	y
6785	sir board school tri assault	0
1905	two guy came stood near start pass cheap comme...	1
3840	ogl facial express comment	1
2396	way home met group boy carri girl tri scream h...	0
7095	incid took place friend around 6 month back fr...	1
tf-idf vectorization
def fn_tfidf(corpus, max_feats = 500, n_gram_range = (1, 1), to_transform = []):  

    kw = dict(max_features = max_feats, ngram_range = n_gram_range)
    tfidf_tranformer = TfidfVectorizer(**kw)
    tfidf_tranformer.fit(corpus)

    to_transform = [corpus] + to_transform  
    transformed_Xs = [tfidf_tranformer.transform(corpus) for corpus in to_transform]  

    return [*transformed_Xs, tfidf_tranformer]
tr_corpus, eval_corpus, ts_corpus = df_tr_.txt.values, df_eval_.txt.values, df_ts_.txt.values
to_transform = [eval_corpus, ts_corpus]

kw = dict(n_gram_range = (1, 1), to_transform = to_transform)

tfidf_tr, tfidf_eval, tfidf_ts, tfidf_transformer = fn_tfidf(tr_corpus, **kw)
Listing out important word
def ten_important_word(tfidf_data):
    
    importance = tfidf_data.A.sum(axis = 0)
    imp_words_idxs = importance.argsort()[::-1]

    imp_words_idxs = imp_words_idxs.flatten()

    imp_words_idxs[:10]
    d = {v:k for k, v in tfidf_transformer.vocabulary_.items()}

    columns = [d[k] for k in imp_words_idxs]
    columns[:10]
    return columns , imp_words_idxs , columns[:10]
columns , imp_words_idxs , ten_word_tr = ten_important_word(tfidf_tr)
ten_word_tr
['comment',
 'touch',
 'boy',
 'guy',
 'girl',
 'man',
 'friend',
 'bu',
 'even',
 'harass']
a,b,ten_word_eval = ten_important_word(tfidf_eval)
ten_word_eval
['comment',
 'boy',
 'touch',
 'guy',
 'man',
 'even',
 'friend',
 'girl',
 'harass',
 'happen']
tfidf_ts_word,b,ten_word_test = ten_important_word(tfidf_ts)
ten_word_test
['comment',
 'touch',
 'boy',
 'man',
 'guy',
 'girl',
 'friend',
 'even',
 'bu',
 'tri']
Word in to feature
df_X_tr = pd.DataFrame(tfidf_tr.A).iloc[:, imp_words_idxs]
df_X_eval = pd.DataFrame(tfidf_eval.A).iloc[:, imp_words_idxs]
df_X_ts = pd.DataFrame(tfidf_ts.A).iloc[:, imp_words_idxs]
df_X_tr.columns, df_X_eval.columns, df_X_ts.columns = columns, columns, columns

df_tr = df_X_tr.assign(y = df_tr_.y.values)
df_eval = df_X_eval.assign(y = df_eval_.y.values)
df_ts = df_X_ts.assign(y = df_ts_.y.values)
df_tr.head(3)
comment	touch	boy	guy	girl	man	friend	bu	even	harass	...	hard	spot	attempt	think	question	signal	may	file	stori	y
0	0.407756	0.0	0.428979	0.000000	0.0	0.0	0.485443	0.0	0.0	0.0	...	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	1
1	0.000000	0.0	0.000000	0.213958	0.0	0.0	0.000000	0.0	0.0	0.0	...	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0
2	0.203483	0.0	0.214074	0.000000	0.0	0.0	0.242251	0.0	0.0	0.0	...	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	1
3 rows × 501 columns

VISUALIZING FEATURE IMPORTANCE:
def fn_feat_importance(df_X_tr, y_tr):

    f_ratios, zzz = f_classif(df_X_tr, y_tr)
    f_ratios = pd.Series(f_ratios, index = df_X_tr.columns)  

    return f_ratios.sort_values(ascending = False)
def fn_corr_matrix(df_X_tr, f_ratios):

    zz = list(f_ratios.index)
    df_corr_mat = df_X_tr.corr(method = 'spearman').abs()
    df_corr = df_corr_mat.loc[zz]
    return df_corr.loc[:, zz]
def fn_filter_feats(df_corr, thresh = 0.8):

    dff = df_corr.copy()
    collect_good_feats = []
    cols = list(dff.columns)
    while True:

        col = cols[0]
        s = dff.loc[:, col]
        ss = s[s.values >= thresh].index
        ss = list(ss)
     
        dff = dff.drop(ss)
        dff = dff.drop(ss, axis = 1)
      
        [cols.remove(i) for i in ss]
        collect_good_feats.append(col)
        if len(cols) < 2:
            break
    return df_corr.loc[collect_good_feats].loc[:, collect_good_feats]
def fn_plot_corr_numerical(f_ratios, df_corr, figsize = (15, 7), fontsize = 15):


    plt.figure(figsize=figsize)

    plt.subplot(1,2,1)
    f_ratios.sort_values().plot(kind = 'barh', alpha = 0.6)
    plt.title('FEATURE_LABEL_CORRELATIONS (anova - f - values)')
    plt.xlabel('Degree_of_Correlation', fontsize = fontsize)
    plt.xticks(fontsize=fontsize-2)
    plt.yticks(fontsize=fontsize-2)

    plt.subplot(1,2,2)        
    heatmap(df_corr, annot=False, cmap="YlGnBu")
    plt.title('FEATURE_FEATURE_CORRELATIONS (SPEARMANS)')
    plt.xticks(fontsize=0)
    plt.yticks(fontsize=0, rotation = 0)
    plt.tight_layout()
    plt.show()
def fn_feat_select_clfn(df_tr, n_feats = None, plot = True,
                             thresh_feat_label = None, 
                             thresh_feat_feat =  None,
                             figsize = (15, 7)):
   
    n_feats = -1 if n_feats == None else n_feats
    
    df_X, y = df_tr.iloc[:, :n_feats], df_tr.iloc[:, -1].values
    f_ratios = fn_feat_importance(df_X, y)
    
    df_corr = fn_corr_matrix(df_X, f_ratios)
   

    if thresh_feat_label == None and thresh_feat_feat == None:
        best_feats = list(f_ratios.index)

    else:
        df_corr = fn_filter_feats(df_corr, thresh = thresh_feat_feat)
       
        f_ratios = f_ratios[df_corr.index]
        f_ratios = f_ratios[f_ratios.values >= thresh_feat_label]
        best_feats = list(f_ratios.index)
       
    
        df_corr = df_corr.loc[best_feats].loc[:, best_feats]
 
    if plot: fn_plot_corr_numerical(f_ratios, df_corr, figsize = figsize)

    return f_ratios, df_corr, best_feats
f_ratios_, df_corr_, best_feats_ = fn_feat_select_clfn(df_tr, 
                                                            thresh_feat_label = 40, 
                                                            thresh_feat_feat =  0.9,
                                                            figsize = (15, 7))

Kde plot for best feature w.r.t target for training dataset
for i in best_feats_:
    plt.figure(figsize=(15,8))
    sns.kdeplot(data=df_tr, x=i, hue="y",multiple="stack")
<Figure size 1080x576 with 0 Axes>
<AxesSubplot:xlabel='comment', ylabel='Density'>
<Figure size 1080x576 with 0 Axes>
<AxesSubplot:xlabel='touch', ylabel='Density'>
<Figure size 1080x576 with 0 Axes>
<AxesSubplot:xlabel='snatch', ylabel='Density'>
<Figure size 1080x576 with 0 Axes>
<AxesSubplot:xlabel='chain', ylabel='Density'>
<Figure size 1080x576 with 0 Axes>
<AxesSubplot:xlabel='rape', ylabel='Density'>
<Figure size 1080x576 with 0 Axes>
<AxesSubplot:xlabel='follow', ylabel='Density'>
<Figure size 1080x576 with 0 Axes>
<AxesSubplot:xlabel='beat', ylabel='Density'>
<Figure size 1080x576 with 0 Axes>
<AxesSubplot:xlabel='grope', ylabel='Density'>
<Figure size 1080x576 with 0 Axes>
<AxesSubplot:xlabel='call', ylabel='Density'>
<Figure size 1080x576 with 0 Axes>
<AxesSubplot:xlabel='stare', ylabel='Density'>
<Figure size 1080x576 with 0 Axes>
<AxesSubplot:xlabel='pass', ylabel='Density'>
<Figure size 1080x576 with 0 Axes>
<AxesSubplot:xlabel='sex', ylabel='Density'>












Logistic regression
TRAINING
def fn_param_grid(param_grid_):
    return ParameterGrid(param_grid_)
param_grid_ = dict(penalty = ['l2', 'l1'],  
                            C = [1e-25, 1e-15, 1e-3, 1e0, 1e15],
                       solver = ['saga'],
                     max_iter = [30_000],
                 random_state = [0])

param_grid = fn_param_grid(param_grid_)
def fn_train_models(X_std, y, model_class, param_grid):

    X = X_std
    trained_models = []
    pbar = ProgressBar()
    for hyp_params in pbar(param_grid):
        trained_model = model_class(**hyp_params).fit(X, y)
        trained_models.append(trained_model)
    trained_models = pd.Series(trained_models)
    return trained_models
from sklearn.linear_model import LogisticRegression

model_class = LogisticRegression
X_tr, y_tr = df_tr.iloc[:, :-1].values, df_tr.iloc[:, -1].values

trained_models = fn_train_models(X_tr, y_tr, model_class, param_grid)
100% |########################################################################|
trained_models
0    LogisticRegression(C=1e-25, max_iter=30000, ra...
1    LogisticRegression(C=1e-25, max_iter=30000, pe...
2    LogisticRegression(C=1e-15, max_iter=30000, ra...
3    LogisticRegression(C=1e-15, max_iter=30000, pe...
4    LogisticRegression(C=0.001, max_iter=30000, ra...
5    LogisticRegression(C=0.001, max_iter=30000, pe...
6    LogisticRegression(max_iter=30000, random_stat...
7    LogisticRegression(max_iter=30000, penalty='l1...
8    LogisticRegression(C=1000000000000000.0, max_i...
9    LogisticRegression(C=1000000000000000.0, max_i...
dtype: object
EVALUATION
def fn_acc_prec_rec(y, y_proba, thresh):    

    y_pred = np.array([1 if i > thresh else 0 for i in y_proba])
    dff = pd.DataFrame().assign(y = y, y_pred = y_pred)
    
    TP_1 = sum([1 for i in dff[dff.y == 1].y_pred if i == 1]) + 1e-6
    FP_1 = sum([1 for i in dff[dff.y == 0].y_pred if i != 0]) + 1e-6
    FN_1 = sum([1 for i in dff[dff.y == 1].y_pred if i != 1]) + 1e-6
    prec_1, rec_1 = TP_1/(TP_1 + FP_1), TP_1/(TP_1 + FN_1)

    TP_0 = sum([1 for i in dff[dff.y == 0].y_pred if i == 0]) + 1e-6
    FP_0 = sum([1 for i in dff[dff.y == 1].y_pred if i != 1]) + 1e-6
    FN_0 = sum([1 for i in dff[dff.y == 0].y_pred if i != 0]) + 1e-6
    prec_0, rec_0 = TP_0/(TP_0 + FP_0), TP_0/(TP_0 + FN_0)
    
    acc = (TP_1 + TP_0)/len(y_pred)  
    
    F1_Measure_0 = (2 * prec_1 * rec_1) / (prec_1 + rec_1)
    F1_Measure_1 = (2 * prec_0 * rec_0) / (prec_0 + rec_0)
    return acc, prec_0, prec_1, rec_0, rec_1,F1_Measure_0,F1_Measure_1
def fn_performance_metrics(y, y_proba, listO_thresholds):
    
    listO_metrics = []
    for thresh in listO_thresholds:

        acc, prec_0, prec_1, rec_0, rec_1,F1_Measure_0 , F1_Measure_1 = fn_acc_prec_rec(y, y_proba, thresh)         
        listO_metrics.append([acc, prec_0, prec_1, rec_0, rec_1,F1_Measure_0 ,F1_Measure_1, thresh])

    df_performance_metrics = pd.DataFrame(np.array(listO_metrics))  
    df_performance_metrics.columns = ['acc', 'prec_0', 'prec_1', 'rec_0', 'rec_1', 'F1_Measure_0','F1_Measure_1' , 'thresh']
    df_performance_metrics.sort_values(by = 'thresh')   
        
    return df_performance_metrics
def fn_pr_rec_tr_eval(y_tr, y_tr_proba, y_eval, y_eval_proba, 
                    class_, listO_thresholds = np.linspace(0, 1, 10).round(2)):

    df_metrics_tr = fn_performance_metrics(y_tr, y_tr_proba, listO_thresholds)
    df_metrics_eval = fn_performance_metrics(y_eval, y_eval_proba, listO_thresholds)

    if class_ == 1:
        tr_prec, tr_rec = df_metrics_tr.prec_1, df_metrics_tr.rec_1
        eval_prec, eval_rec = df_metrics_eval.prec_1, df_metrics_eval.rec_1
    if class_ == 0:
        tr_prec, tr_rec = df_metrics_tr.prec_0, df_metrics_tr.rec_0
        eval_prec, eval_rec = df_metrics_eval.prec_0, df_metrics_eval.rec_0

    acc_tr, acc_eval = df_metrics_tr.acc, df_metrics_eval.acc
    
    subplot_grid = (1, 2)
    fig, axes = plt.subplots(*subplot_grid, figsize=(10, 4), sharey = True)
    axes = axes.ravel()

    axes[0].plot(listO_thresholds, tr_prec, label = 'precision')
    axes[0].plot(listO_thresholds, tr_rec, label = 'recall')
    axes[0].plot(listO_thresholds, acc_tr, linestyle = '--', label = 'accuracy')
    axes[0].set_xlabel('thresholds')
    axes[0].set_title('TRAIN - ' + 'class_' + str(class_))
    
    axes[1].plot(listO_thresholds, eval_prec, label = 'precision')
    axes[1].plot(listO_thresholds, eval_rec, label = 'recall')
    axes[1].plot(listO_thresholds, acc_eval, linestyle = '--', label = 'accuracy')
    axes[1].set_xlabel('thresholds')
    axes[1].set_title('EVAL - ' + 'class_' + str(class_))

    plt.legend()
    plt.tight_layout()
for i in range(trained_models.shape[0]):
    X_eval, y_eval = df_eval.iloc[:, :-1].values, df_eval.iloc[:, -1].values
    y_tr_proba = trained_models[i].predict_proba(X_tr)[:, 1]
    y_eval_proba = trained_models[i].predict_proba(X_eval)[:, 1]

    fn_pr_rec_tr_eval(y_tr, y_tr_proba, y_eval, y_eval_proba, class_ = 1)










TESTING
def fn_pred_proba(model, X):
    if hasattr(model, "predict_proba"):
        prob_pos = model.predict_proba(X)[:, 1]
    else:  # For model without pred_proba
        prob_pos = model.decision_function(X)
        prob_pos = (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())

    return prob_pos
from sklearn.metrics import roc_curve, roc_auc_score
def fn_test_model_binary_clf(df_Xy_, model_, threshold_class_1 = 0.5):
    
    df, model = df_Xy_, model_
    X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values.ravel()
    y_proba =  fn_pred_proba(model, X)

    logloss = log_loss(y, y_proba, labels=model_.classes_)
    acc, prec_0, prec_1, rec_0, rec_1,F1_Measure_0,F1_Measure_1 = fn_acc_prec_rec(y, y_proba, threshold_class_1) 
    
    df = pd.DataFrame().assign(prec = (prec_0, prec_1), rec = (rec_0, rec_1), f1 =(F1_Measure_0,F1_Measure_1) )
    df.index = ['class_' + str(i) for i in range(len(df))]
    
    print(f'LOGLOSS : {round(logloss, 4)}')
    print(f'ACCURACY: {round(acc, 3)}')
    print()

    false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(y, y_proba)
    print('roc_auc_score: ', roc_auc_score(y, y_proba))
    plt.subplots(1, figsize=(5,5))
    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate1, true_positive_rate1)
    plt.plot([0, 1], ls="--")
    plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    return df.round(3)
for i in range(trained_models.shape[0]):
    print('---------------------------------------------------------------------------------------------', i )
    model_ = trained_models[i]
    fn_test_model_binary_clf(df_ts, model_, threshold_class_1 = 0.5)
    print('---------------------------------------------------------------------------------------------  End of' , i)
--------------------------------------------------------------------------------------------- 0
LOGLOSS : 0.6931
ACCURACY: 0.313

roc_auc_score:  0.5

prec	rec	f1
class_0	0.313	1.0	0.000
class_1	0.500	0.0	0.477
---------------------------------------------------------------------------------------------  End of 0
--------------------------------------------------------------------------------------------- 1
LOGLOSS : 0.6377
ACCURACY: 0.687

roc_auc_score:  0.5

prec	rec	f1
class_0	0.500	0.0	0.814
class_1	0.687	1.0	0.000
---------------------------------------------------------------------------------------------  End of 1
--------------------------------------------------------------------------------------------- 2
LOGLOSS : 0.6931
ACCURACY: 0.687

roc_auc_score:  0.7963381038769065

prec	rec	f1
class_0	0.500	0.0	0.814
class_1	0.687	1.0	0.000
---------------------------------------------------------------------------------------------  End of 2
--------------------------------------------------------------------------------------------- 3
LOGLOSS : 0.6377
ACCURACY: 0.687

roc_auc_score:  0.5

prec	rec	f1
class_0	0.500	0.0	0.814
class_1	0.687	1.0	0.000
---------------------------------------------------------------------------------------------  End of 3
--------------------------------------------------------------------------------------------- 4
LOGLOSS : 0.6177
ACCURACY: 0.687

roc_auc_score:  0.8579531456471589

prec	rec	f1
class_0	0.500	0.0	0.814
class_1	0.687	1.0	0.000
---------------------------------------------------------------------------------------------  End of 4
--------------------------------------------------------------------------------------------- 5
LOGLOSS : 0.6376
ACCURACY: 0.687

roc_auc_score:  0.5

prec	rec	f1
class_0	0.500	0.0	0.814
class_1	0.687	1.0	0.000
---------------------------------------------------------------------------------------------  End of 5
--------------------------------------------------------------------------------------------- 6
LOGLOSS : 0.4234
ACCURACY: 0.809

roc_auc_score:  0.8661862527716186

prec	rec	f1
class_0	0.754	0.579	0.868
class_1	0.826	0.914	0.655
---------------------------------------------------------------------------------------------  End of 6
--------------------------------------------------------------------------------------------- 7
LOGLOSS : 0.4175
ACCURACY: 0.818

roc_auc_score:  0.8693486976192076

prec	rec	f1
class_0	0.747	0.634	0.872
class_1	0.844	0.902	0.686
---------------------------------------------------------------------------------------------  End of 7
--------------------------------------------------------------------------------------------- 8
LOGLOSS : 0.4961
ACCURACY: 0.786

roc_auc_score:  0.8387052341597796

prec	rec	f1
class_0	0.675	0.608	0.847
class_1	0.829	0.867	0.639
---------------------------------------------------------------------------------------------  End of 8
--------------------------------------------------------------------------------------------- 9
LOGLOSS : 0.4961
ACCURACY: 0.786

roc_auc_score:  0.8387052341597796

prec	rec	f1
class_0	0.675	0.608	0.847
class_1	0.829	0.867	0.639
---------------------------------------------------------------------------------------------  End of 9
Random Forest Classifier
Training
param_grid_r = dict(n_estimators = [50,1000],  
                            min_samples_leaf = [2,4])

param_grid_r = fn_param_grid(param_grid_r)
param_grid_r
<sklearn.model_selection._search.ParameterGrid at 0x2125d536070>
EVALUATION
from sklearn.ensemble import RandomForestClassifier
model_class_r = RandomForestClassifier

trained_models_r = fn_train_models(X_tr, y_tr, model_class_r, param_grid_r)
trained_models_r
100% |########################################################################|
0    (DecisionTreeClassifier(max_features='sqrt', m...
1    (DecisionTreeClassifier(max_features='sqrt', m...
2    (DecisionTreeClassifier(max_features='sqrt', m...
3    (DecisionTreeClassifier(max_features='sqrt', m...
dtype: object
for i in range(trained_models_r.shape[0]):
    X_eval, y_eval = df_eval.iloc[:, :-1].values, df_eval.iloc[:, -1].values
    y_tr_proba = trained_models_r[i].predict_proba(X_tr)[:, 1]
    y_eval_proba = trained_models_r[i].predict_proba(X_eval)[:, 1]

    fn_pr_rec_tr_eval(y_tr, y_tr_proba, y_eval, y_eval_proba, class_ = 1)




TESTING
for i in range(trained_models_r.shape[0]):
    print('---------------------------------------------------------------------------------------------',i)
    model_r = trained_models_r[i]
    fn_test_model_binary_clf(df_ts, model_r, threshold_class_1 = 0.6)
    print('---------------------------------------------------------------------------------------------')
--------------------------------------------------------------------------------------------- 0
LOGLOSS : 0.4176
ACCURACY: 0.802

roc_auc_score:  0.867297139913548

prec	rec	f1
class_0	0.669	0.729	0.853
class_1	0.871	0.835	0.698
---------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------- 1
LOGLOSS : 0.4156
ACCURACY: 0.804

roc_auc_score:  0.8693778136128469

prec	rec	f1
class_0	0.667	0.747	0.854
class_1	0.878	0.830	0.705
---------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------- 2
LOGLOSS : 0.4214
ACCURACY: 0.804

roc_auc_score:  0.8685804833254945

prec	rec	f1
class_0	0.669	0.741	0.854
class_1	0.876	0.833	0.703
---------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------- 3
LOGLOSS : 0.4202
ACCURACY: 0.801

roc_auc_score:  0.8708268942193554

prec	rec	f1
class_0	0.661	0.747	0.851
class_1	0.878	0.825	0.701
---------------------------------------------------------------------------------------------
GradientBoostingClassifier
Training
from sklearn.ensemble import GradientBoostingClassifier
param_grid_gbdt = dict(n_estimators = [50,100],  
                            min_samples_leaf = [2,4],
                           learning_rate =[0.01,0.1,1,10,100])

param_grid_g = fn_param_grid(param_grid_gbdt)
param_grid_g
<sklearn.model_selection._search.ParameterGrid at 0x2124e1ccb50>
EVALUATION
model_class_gbdt = GradientBoostingClassifier
trained_models_gbdt = fn_train_models(X_tr, y_tr, model_class_gbdt, param_grid_g)
trained_models_gbdt
100% |########################################################################|
0     ([DecisionTreeRegressor(criterion='friedman_ms...
1     ([DecisionTreeRegressor(criterion='friedman_ms...
2     ([DecisionTreeRegressor(criterion='friedman_ms...
3     ([DecisionTreeRegressor(criterion='friedman_ms...
4     ([DecisionTreeRegressor(criterion='friedman_ms...
5     ([DecisionTreeRegressor(criterion='friedman_ms...
6     ([DecisionTreeRegressor(criterion='friedman_ms...
7     ([DecisionTreeRegressor(criterion='friedman_ms...
8     ([DecisionTreeRegressor(criterion='friedman_ms...
9     ([DecisionTreeRegressor(criterion='friedman_ms...
10    ([DecisionTreeRegressor(criterion='friedman_ms...
11    ([DecisionTreeRegressor(criterion='friedman_ms...
12    ([DecisionTreeRegressor(criterion='friedman_ms...
13    ([DecisionTreeRegressor(criterion='friedman_ms...
14    ([DecisionTreeRegressor(criterion='friedman_ms...
15    ([DecisionTreeRegressor(criterion='friedman_ms...
16    ([DecisionTreeRegressor(criterion='friedman_ms...
17    ([DecisionTreeRegressor(criterion='friedman_ms...
18    ([DecisionTreeRegressor(criterion='friedman_ms...
19    ([DecisionTreeRegressor(criterion='friedman_ms...
dtype: object
for i in range(trained_models_gbdt.shape[0]):
    X_eval, y_eval = df_eval.iloc[:, :-1].values, df_eval.iloc[:, -1].values
    y_tr_proba = trained_models_gbdt[i].predict_proba(X_tr)[:, 1]
    y_eval_proba = trained_models_gbdt[i].predict_proba(X_eval)[:, 1]

    fn_pr_rec_tr_eval(y_tr, y_tr_proba, y_eval, y_eval_proba, class_ = 1)




















TESTING
for i in range(trained_models_gbdt.shape[0]):
    print('--------------------------------------------------------------------------------------------- ', i)
    model_gbdt = trained_models_gbdt[i]
    fn_test_model_binary_clf(df_ts, model_gbdt, threshold_class_1 = 0.6)
    print('---------------------------------------------------------------------------------------------')
---------------------------------------------------------------------------------------------  0
LOGLOSS : 0.5546
ACCURACY: 0.728

roc_auc_score:  0.760112208560102

prec	rec	f1
class_0	0.883	0.151	0.833
class_1	0.719	0.991	0.258
---------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------  1
LOGLOSS : 0.524
ACCURACY: 0.745

roc_auc_score:  0.794470200900356

prec	rec	f1
class_0	0.753	0.277	0.838
class_1	0.744	0.959	0.405
---------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------  2
LOGLOSS : 0.554
ACCURACY: 0.729

roc_auc_score:  0.7624840421957938

prec	rec	f1
class_0	0.895	0.151	0.834
class_1	0.719	0.992	0.258
---------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------  3
LOGLOSS : 0.5231
ACCURACY: 0.746

roc_auc_score:  0.7954511859168178

prec	rec	f1
class_0	0.758	0.277	0.838
class_1	0.745	0.960	0.406
---------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------  4
LOGLOSS : 0.4566
ACCURACY: 0.811

roc_auc_score:  0.8535868664471767

prec	rec	f1
class_0	0.726	0.634	0.866
class_1	0.842	0.891	0.677
---------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------  5
LOGLOSS : 0.4361
ACCURACY: 0.806

roc_auc_score:  0.8621368899639409

prec	rec	f1
class_0	0.701	0.665	0.861
class_1	0.851	0.871	0.683
---------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------  6
LOGLOSS : 0.4563
ACCURACY: 0.81

roc_auc_score:  0.8548377343277564

prec	rec	f1
class_0	0.719	0.645	0.865
class_1	0.846	0.885	0.680
---------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------  7
LOGLOSS : 0.4355
ACCURACY: 0.813

roc_auc_score:  0.8625769894062577

prec	rec	f1
class_0	0.711	0.681	0.865
class_1	0.857	0.874	0.695
---------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------  8
LOGLOSS : 0.5116
ACCURACY: 0.789

roc_auc_score:  0.8346480324307375

prec	rec	f1
class_0	0.666	0.654	0.847
class_1	0.844	0.851	0.660
---------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------  9
LOGLOSS : 0.5218
ACCURACY: 0.792

roc_auc_score:  0.8299614773007234

prec	rec	f1
class_0	0.670	0.661	0.849
class_1	0.846	0.852	0.665
---------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------  10
LOGLOSS : 0.4951
ACCURACY: 0.784

roc_auc_score:  0.831195547492665

prec	rec	f1
class_0	0.658	0.645	0.844
class_1	0.840	0.847	0.652
---------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------  11
LOGLOSS : 0.5915
ACCURACY: 0.774

roc_auc_score:  0.8190396201482675

prec	rec	f1
class_0	0.636	0.652	0.835
class_1	0.840	0.830	0.644
---------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------  12
LOGLOSS : 21.8596
ACCURACY: 0.367

roc_auc_score:  0.2834688346883469

prec	rec	f1
class_0	0.052	0.060	0.524
class_1	0.542	0.507	0.056
---------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------  13
LOGLOSS : 21.8596
ACCURACY: 0.367

roc_auc_score:  0.2834688346883469

prec	rec	f1
class_0	0.052	0.060	0.524
class_1	0.542	0.507	0.056
---------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------  14
LOGLOSS : 21.8356
ACCURACY: 0.368

roc_auc_score:  0.2839738851933974

prec	rec	f1
class_0	0.053	0.060	0.525
class_1	0.543	0.508	0.056
---------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------  15
LOGLOSS : 21.8356
ACCURACY: 0.368

roc_auc_score:  0.2839738851933974

prec	rec	f1
class_0	0.053	0.060	0.525
class_1	0.543	0.508	0.056
---------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------  16
LOGLOSS : 21.7877
ACCURACY: 0.369

roc_auc_score:  0.284380389258438

prec	rec	f1
class_0	0.051	0.058	0.527
class_1	0.544	0.511	0.054
---------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------  17
LOGLOSS : 21.7877
ACCURACY: 0.369

roc_auc_score:  0.284380389258438

prec	rec	f1
class_0	0.051	0.058	0.527
class_1	0.544	0.511	0.054
---------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------  18
LOGLOSS : 21.8117
ACCURACY: 0.368

roc_auc_score:  0.28447893569844784

prec	rec	f1
class_0	0.053	0.060	0.526
class_1	0.543	0.509	0.056
---------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------  19
LOGLOSS : 21.8117
ACCURACY: 0.368

roc_auc_score:  0.28447893569844784

prec	rec	f1
class_0	0.053	0.060	0.526
class_1	0.543	0.509	0.056
---------------------------------------------------------------------------------------------
 
Oversampling with SMOTE Oversampling
We will use SMOTE Oversampling method to handle the class imbalance
Plot of distribution of the two classes before Oversampling
count = [sum(y_tr == 1), sum(y_tr == 0)]
label = ['1', '0']
fig = plt.figure(figsize = (10, 5))
plt.title(" Distribution of class labels")
plt.xlabel("----------Classes--------")
plt.ylabel("Count")
sns.barplot(label,count) 
plt.show()
Text(0.5, 1.0, ' Distribution of class labels')
Text(0.5, 0, '----------Classes--------')
Text(0, 0.5, 'Count')
<AxesSubplot:title={'center':' Distribution of class labels'}, xlabel='----------Classes--------', ylabel='Count'>

Plot of distribution of the two classes After Oversampling
# X_tr, y_tr = df_tr.iloc[:, :-1].values, df_tr.iloc[:, -1].values
from imblearn.over_sampling import SMOTE
oversample = SMOTE()
X, y = oversample.fit_resample(X_tr, y_tr)
# counter = Counter(y)
# print(counter)

count = [sum(y == 1), sum(y == 0)]
label = ['1', '0']
fig = plt.figure(figsize = (10, 5))
plt.title(" Distribution of class labels")
plt.xlabel("----------Classes--------")
plt.ylabel("Count")
sns.barplot(label,count) 
plt.show()
Text(0.5, 1.0, ' Distribution of class labels')
Text(0.5, 0, '----------Classes--------')
Text(0, 0.5, 'Count')
<AxesSubplot:title={'center':' Distribution of class labels'}, xlabel='----------Classes--------', ylabel='Count'>

Logistic regression
trained_models_smote = fn_train_models(X, y, model_class, param_grid)
100% |########################################################################|
trained_models_smote
0    LogisticRegression(C=1e-25, max_iter=30000, ra...
1    LogisticRegression(C=1e-25, max_iter=30000, pe...
2    LogisticRegression(C=1e-15, max_iter=30000, ra...
3    LogisticRegression(C=1e-15, max_iter=30000, pe...
4    LogisticRegression(C=0.001, max_iter=30000, ra...
5    LogisticRegression(C=0.001, max_iter=30000, pe...
6    LogisticRegression(max_iter=30000, random_stat...
7    LogisticRegression(max_iter=30000, penalty='l1...
8    LogisticRegression(C=1000000000000000.0, max_i...
9    LogisticRegression(C=1000000000000000.0, max_i...
dtype: object
for i in range(trained_models.shape[0]):
    X_eval, y_eval = df_eval.iloc[:, :-1].values, df_eval.iloc[:, -1].values
    y_tr_proba = trained_models_smote[i].predict_proba(X)[:, 1]
    y_eval_proba = trained_models_smote[i].predict_proba(X_eval)[:, 1]

    fn_pr_rec_tr_eval(y, y_tr_proba, y_eval, y_eval_proba, class_ = 1)










for i in range(trained_models_smote.shape[0]):
    print('---------------------------------------------------------------------------------------------', i )
    model_ = trained_models_smote[i]
    fn_test_model_binary_clf(df_ts, model_, threshold_class_1 = 0.5)
    print('---------------------------------------------------------------------------------------------  End of' , i)
--------------------------------------------------------------------------------------------- 0
LOGLOSS : 0.6931
ACCURACY: 0.313

roc_auc_score:  0.5

prec	rec	f1
class_0	0.313	1.0	0.000
class_1	0.500	0.0	0.477
---------------------------------------------------------------------------------------------  End of 0
--------------------------------------------------------------------------------------------- 1
LOGLOSS : 0.8412
ACCURACY: 0.313

roc_auc_score:  0.5

prec	rec	f1
class_0	0.313	1.0	0.000
class_1	0.500	0.0	0.477
---------------------------------------------------------------------------------------------  End of 1
--------------------------------------------------------------------------------------------- 2
LOGLOSS : 0.6931
ACCURACY: 0.313

roc_auc_score:  0.8496147730072342

prec	rec	f1
class_0	0.313	1.0	0.000
class_1	0.500	0.0	0.477
---------------------------------------------------------------------------------------------  End of 2
--------------------------------------------------------------------------------------------- 3
LOGLOSS : 0.8412
ACCURACY: 0.313

roc_auc_score:  0.5

prec	rec	f1
class_0	0.313	1.0	0.000
class_1	0.500	0.0	0.477
---------------------------------------------------------------------------------------------  End of 3
--------------------------------------------------------------------------------------------- 4
LOGLOSS : 0.6863
ACCURACY: 0.751

roc_auc_score:  0.8500963067481915

prec	rec	f1
class_0	0.571	0.818	0.799
class_1	0.897	0.720	0.673
---------------------------------------------------------------------------------------------  End of 4
--------------------------------------------------------------------------------------------- 5
LOGLOSS : 0.8413
ACCURACY: 0.313

roc_auc_score:  0.5

prec	rec	f1
class_0	0.313	1.0	0.000
class_1	0.500	0.0	0.477
---------------------------------------------------------------------------------------------  End of 5
--------------------------------------------------------------------------------------------- 6
LOGLOSS : 0.4533
ACCURACY: 0.788

roc_auc_score:  0.8579688234898877

prec	rec	f1
class_0	0.636	0.754	0.839
class_1	0.877	0.803	0.690
---------------------------------------------------------------------------------------------  End of 6
--------------------------------------------------------------------------------------------- 7
LOGLOSS : 0.4591
ACCURACY: 0.777

roc_auc_score:  0.8580416134739859

prec	rec	f1
class_0	0.612	0.780	0.827
class_1	0.886	0.775	0.686
---------------------------------------------------------------------------------------------  End of 7
--------------------------------------------------------------------------------------------- 8
LOGLOSS : 0.5505
ACCURACY: 0.767

roc_auc_score:  0.8278964814441532

prec	rec	f1
class_0	0.613	0.692	0.825
class_1	0.851	0.801	0.650
---------------------------------------------------------------------------------------------  End of 8
--------------------------------------------------------------------------------------------- 9
LOGLOSS : 0.5505
ACCURACY: 0.767

roc_auc_score:  0.8278964814441532

prec	rec	f1
class_0	0.613	0.692	0.825
class_1	0.851	0.801	0.650
---------------------------------------------------------------------------------------------  End of 9
Random Forest Classifier
trained_models_r_smote = fn_train_models(X, y, model_class_r, param_grid_r)
trained_models_r_smote
100% |########################################################################|
0    (DecisionTreeClassifier(max_features='sqrt', m...
1    (DecisionTreeClassifier(max_features='sqrt', m...
2    (DecisionTreeClassifier(max_features='sqrt', m...
3    (DecisionTreeClassifier(max_features='sqrt', m...
dtype: object
for i in range(trained_models_r.shape[0]):
    y_tr_proba = trained_models_r_smote[i].predict_proba(X)[:, 1]
    y_eval_proba = trained_models_r_smote[i].predict_proba(X_eval)[:, 1]
    fn_pr_rec_tr_eval(y, y_tr_proba, y_eval, y_eval_proba, class_ = 1)




for i in range(trained_models_r_smote.shape[0]):
    print('---------------------------------------------------------------------------------------------', i )
    _smote = trained_models_r_smote[i]
    fn_test_model_binary_clf(df_ts, _smote, threshold_class_1 = 0.7)
    print('---------------------------------------------------------------------------------------------  End of' , i)
--------------------------------------------------------------------------------------------- 0
LOGLOSS : 0.4382
ACCURACY: 0.734

roc_auc_score:  0.8569228874106922

prec	rec	f1
class_0	0.546	0.882	0.774
class_1	0.926	0.666	0.675
---------------------------------------------------------------------------------------------  End of 0
--------------------------------------------------------------------------------------------- 1
LOGLOSS : 0.438
ACCURACY: 0.735

roc_auc_score:  0.857440256220744

prec	rec	f1
class_0	0.547	0.887	0.775
class_1	0.928	0.666	0.677
---------------------------------------------------------------------------------------------  End of 1
--------------------------------------------------------------------------------------------- 2
LOGLOSS : 0.4502
ACCURACY: 0.717

roc_auc_score:  0.8543394028981612

prec	rec	f1
class_0	0.528	0.900	0.755
class_1	0.933	0.633	0.666
---------------------------------------------------------------------------------------------  End of 2
--------------------------------------------------------------------------------------------- 3
LOGLOSS : 0.4506
ACCURACY: 0.718

roc_auc_score:  0.8547257497368362

prec	rec	f1
class_0	0.529	0.900	0.756
class_1	0.933	0.635	0.667
---------------------------------------------------------------------------------------------  End of 3
GradientBoostingClassifier
trained_models_gbdt_smote = fn_train_models(X, y, model_class_gbdt, param_grid_g)
trained_models_gbdt_smote
100% |########################################################################|
0     ([DecisionTreeRegressor(criterion='friedman_ms...
1     ([DecisionTreeRegressor(criterion='friedman_ms...
2     ([DecisionTreeRegressor(criterion='friedman_ms...
3     ([DecisionTreeRegressor(criterion='friedman_ms...
4     ([DecisionTreeRegressor(criterion='friedman_ms...
5     ([DecisionTreeRegressor(criterion='friedman_ms...
6     ([DecisionTreeRegressor(criterion='friedman_ms...
7     ([DecisionTreeRegressor(criterion='friedman_ms...
8     ([DecisionTreeRegressor(criterion='friedman_ms...
9     ([DecisionTreeRegressor(criterion='friedman_ms...
10    ([DecisionTreeRegressor(criterion='friedman_ms...
11    ([DecisionTreeRegressor(criterion='friedman_ms...
12    ([DecisionTreeRegressor(criterion='friedman_ms...
13    ([DecisionTreeRegressor(criterion='friedman_ms...
14    ([DecisionTreeRegressor(criterion='friedman_ms...
15    ([DecisionTreeRegressor(criterion='friedman_ms...
16    ([DecisionTreeRegressor(criterion='friedman_ms...
17    ([DecisionTreeRegressor(criterion='friedman_ms...
18    ([DecisionTreeRegressor(criterion='friedman_ms...
19    ([DecisionTreeRegressor(criterion='friedman_ms...
dtype: object
for i in range(trained_models_gbdt_smote.shape[0]):
    X_eval, y_eval = df_eval.iloc[:, :-1].values, df_eval.iloc[:, -1].values
    y_tr_proba = trained_models_gbdt_smote[i].predict_proba(X)[:, 1]
    y_eval_proba = trained_models_gbdt_smote[i].predict_proba(X_eval)[:, 1]

    fn_pr_rec_tr_eval(y, y_tr_proba, y_eval, y_eval_proba, class_ = 1)




















for i in range(trained_models_gbdt_smote.shape[0]):
    print('---------------------------------------------------------------------------------------------', i )
    _smote = trained_models_gbdt_smote[i]
    fn_test_model_binary_clf(df_ts, _smote, threshold_class_1 = 0.7)
    print('---------------------------------------------------------------------------------------------  End of' , i)
--------------------------------------------------------------------------------------------- 0
LOGLOSS : 0.6071
ACCURACY: 0.313

roc_auc_score:  0.7711852449103003

prec	rec	f1
class_0	0.313	1.0	0.000
class_1	0.500	0.0	0.477
---------------------------------------------------------------------------------------------  End of 0
--------------------------------------------------------------------------------------------- 1
LOGLOSS : 0.5708
ACCURACY: 0.613

roc_auc_score:  0.7826591860959932

prec	rec	f1
class_0	0.444	0.949	0.620
class_1	0.952	0.460	0.605
---------------------------------------------------------------------------------------------  End of 1
--------------------------------------------------------------------------------------------- 2
LOGLOSS : 0.6072
ACCURACY: 0.313

roc_auc_score:  0.7714327308562341

prec	rec	f1
class_0	0.313	1.0	0.000
class_1	0.500	0.0	0.477
---------------------------------------------------------------------------------------------  End of 2
--------------------------------------------------------------------------------------------- 3
LOGLOSS : 0.5709
ACCURACY: 0.612

roc_auc_score:  0.7766411341799367

prec	rec	f1
class_0	0.444	0.951	0.618
class_1	0.954	0.458	0.606
---------------------------------------------------------------------------------------------  End of 3
--------------------------------------------------------------------------------------------- 4
LOGLOSS : 0.4979
ACCURACY: 0.629

roc_auc_score:  0.8467658850142219

prec	rec	f1
class_0	0.456	0.945	0.643
class_1	0.951	0.486	0.615
---------------------------------------------------------------------------------------------  End of 4
--------------------------------------------------------------------------------------------- 5
LOGLOSS : 0.473
ACCURACY: 0.679

roc_auc_score:  0.8568960111088715

prec	rec	f1
class_0	0.493	0.929	0.707
class_1	0.946	0.565	0.644
---------------------------------------------------------------------------------------------  End of 5
--------------------------------------------------------------------------------------------- 6
LOGLOSS : 0.4985
ACCURACY: 0.631

roc_auc_score:  0.8471051983247105

prec	rec	f1
class_0	0.457	0.947	0.644
class_1	0.953	0.487	0.616
---------------------------------------------------------------------------------------------  End of 6
--------------------------------------------------------------------------------------------- 7
LOGLOSS : 0.4741
ACCURACY: 0.677

roc_auc_score:  0.8568444981970481

prec	rec	f1
class_0	0.492	0.927	0.706
class_1	0.944	0.564	0.643
---------------------------------------------------------------------------------------------  End of 7
--------------------------------------------------------------------------------------------- 8
LOGLOSS : 0.5886
ACCURACY: 0.734

roc_auc_score:  0.8134045555331586

prec	rec	f1
class_0	0.551	0.816	0.783
class_1	0.893	0.697	0.658
---------------------------------------------------------------------------------------------  End of 8
--------------------------------------------------------------------------------------------- 9
LOGLOSS : 0.6389
ACCURACY: 0.727

roc_auc_score:  0.8105444690810545

prec	rec	f1
class_0	0.546	0.767	0.781
class_1	0.870	0.709	0.638
---------------------------------------------------------------------------------------------  End of 9
--------------------------------------------------------------------------------------------- 10
LOGLOSS : 0.5581
ACCURACY: 0.731

roc_auc_score:  0.8124728437367019

prec	rec	f1
class_0	0.548	0.803	0.781
class_1	0.886	0.699	0.652
---------------------------------------------------------------------------------------------  End of 10
--------------------------------------------------------------------------------------------- 11
LOGLOSS : 0.5915
ACCURACY: 0.727

roc_auc_score:  0.810961051759278

prec	rec	f1
class_0	0.544	0.787	0.778
class_1	0.878	0.699	0.643
---------------------------------------------------------------------------------------------  End of 11
--------------------------------------------------------------------------------------------- 12
LOGLOSS : 19.1989
ACCURACY: 0.444

roc_auc_score:  0.5888149790588815

prec	rec	f1
class_0	0.358	0.976	0.333
class_1	0.948	0.202	0.523
---------------------------------------------------------------------------------------------  End of 12
--------------------------------------------------------------------------------------------- 13
LOGLOSS : 19.1989
ACCURACY: 0.444

roc_auc_score:  0.5888149790588815

prec	rec	f1
class_0	0.358	0.976	0.333
class_1	0.948	0.202	0.523
---------------------------------------------------------------------------------------------  End of 13
--------------------------------------------------------------------------------------------- 14
LOGLOSS : 19.1989
ACCURACY: 0.444

roc_auc_score:  0.5888149790588815

prec	rec	f1
class_0	0.358	0.976	0.333
class_1	0.948	0.202	0.523
---------------------------------------------------------------------------------------------  End of 14
--------------------------------------------------------------------------------------------- 15
LOGLOSS : 19.1989
ACCURACY: 0.444

roc_auc_score:  0.5888149790588815

prec	rec	f1
class_0	0.358	0.976	0.333
class_1	0.948	0.202	0.523
---------------------------------------------------------------------------------------------  End of 15
--------------------------------------------------------------------------------------------- 16
LOGLOSS : 17.9048
ACCURACY: 0.482

roc_auc_score:  0.3668021680216802

prec	rec	f1
class_0	0.077	0.060	0.641
class_1	0.611	0.674	0.067
---------------------------------------------------------------------------------------------  End of 16
--------------------------------------------------------------------------------------------- 17
LOGLOSS : 17.9048
ACCURACY: 0.482

roc_auc_score:  0.3668021680216802

prec	rec	f1
class_0	0.077	0.060	0.641
class_1	0.611	0.674	0.067
---------------------------------------------------------------------------------------------  End of 17
--------------------------------------------------------------------------------------------- 18
LOGLOSS : 17.9048
ACCURACY: 0.482

roc_auc_score:  0.3668021680216802

prec	rec	f1
class_0	0.077	0.060	0.641
class_1	0.611	0.674	0.067
---------------------------------------------------------------------------------------------  End of 18
--------------------------------------------------------------------------------------------- 19
LOGLOSS : 17.9048
ACCURACY: 0.482

roc_auc_score:  0.3668021680216802

prec	rec	f1
class_0	0.077	0.060	0.641
class_1	0.611	0.674	0.067
---------------------------------------------------------------------------------------------  End of 19
Best model
from sklearn.metrics import roc_curve, roc_auc_score
def roc_compression(modelName , df_Xy_, model_, threshold_class_1 = 0.5):
    
    df, model = df_Xy_, model_
    X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values.ravel()
    y_proba =  fn_pred_proba(model, X)
    plt.title('Receiver Operating Characteristic')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(y, y_proba)
    auc = round(roc_auc_score(y, y_proba), 4)
    print('roc_auc_score: ', roc_auc_score(y, y_proba))
    plt.plot(false_positive_rate1,true_positive_rate1,label=modelName+str(auc))
    plt.legend()
ROC compression - default
roc_compression("Logistic Regression, AUC=" , df_ts, trained_models[7], threshold_class_1 = 0.7)
roc_compression( "Random Forest, AUC=" , df_ts, model_rf, threshold_class_1 = 0.7)
roc_compression("Gradient Boosting, AUC=",df_ts, trained_models_gbdt[7], threshold_class_1 = 0.7)
roc_auc_score:  0.8693486976192076
roc_auc_score:  0.8722110237631302
roc_auc_score:  0.8625769894062577

roc_compression("Logistic Regression, AUC=" , df_ts, trained_models_smote[6], threshold_class_1 = 0.7)
roc_compression( "Random Forest, AUC=" , df_ts, trained_models_r_smote[3], threshold_class_1 = 0.7)
roc_compression("Gradient Boosting, AUC=",df_ts, trained_models_gbdt_smote[5], threshold_class_1 = 0.7)
roc_auc_score:  0.8579688234898877
roc_auc_score:  0.8547257497368362
roc_auc_score:  0.8568960111088715

Prediction
df_data.sample(5)
Description	y
6314	I was walking home from the stage and a man I ...	1
298	One day a friend of mine was just moving along...	0
5038	One guy standing on the streets threw some wat...	0
6201	Once I was standing at a bus stop. Another gir...	1
75	The incident took place in a bus when two guys...	0
df_data.Description[4210]
'teacher beats pupils'
import joblib
model_tfidf = joblib.load('sexual-harasment-tf-idf')
model_rf = joblib.load('sexual-harasment-model')
input_ = ["my friend was being sexually harassed by someone in a high position. finally she had to leave the job but later she complained to the hr. the hr started to organize workshops after that"]
preprocess_text = fn_preprocess_text(input_)
preprocess_text
input_data_features = model_tfidf.transform([preprocess_text])
prediction = model_rf.predict(input_data_features)
prediction_proba = model_rf.predict_proba(input_data_features)
prediction[0] , prediction_proba[[0]]
'friend sexual harass someon high posit final leav job later complain hr hr start organ workshop'
(1, array([[0.40166713, 0.59833287]]))
preprocess_text = fn_preprocess_text(input_)
preprocess_text
input_data_features = model_tfidf.transform([preprocess_text])
prediction = trained_models[7].predict(input_data_features)
prediction_proba = trained_models[7].predict_proba(input_data_features)
prediction[0] , prediction_proba[[0]]
'friend sexual harass someon high posit final leav job later complain hr hr start organ workshop'
(1, array([[0.49871284, 0.50128716]]))
preprocess_text = fn_preprocess_text(input_)
preprocess_text
input_data_features = model_tfidf.transform([preprocess_text])
prediction = trained_models_smote[7].predict(input_data_features)
prediction_proba = trained_models_smote[7].predict_proba(input_data_features)
prediction[0] , prediction_proba[[0]]
'friend sexual harass someon high posit final leav job later complain hr hr start organ workshop'
(0, array([[0.8444601, 0.1555399]]))
Observation
We have successfully predicted the “Default Cases”, We have achieved an accuracy of about 80.04%, Log loss of 41.88% and recall of class 0 and 1 as 74% and 82% a roc score of about 87.22 %, Thus, it is concluded that we get better results using Random Forest model trained on without sampled data.

 
