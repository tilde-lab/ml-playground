import pandas as pd
from mpds_client import MPDSDataRetrieval, MPDSDataTypes

data = "harmunite hatrurite boggsite faujasite-Mg gottardiite mutinaite parthéite paulingite-Ca perlialite bernalite ammonioalunite \
meta-aluminate schwertmannite hazenite balyakinite carlfriesite mroseite clearcreekite hanawaltite donharrisite birchite drobecite \
lazaridisite Swedenborgite alburnite ichnusaite alsakharovite-Zn carbokentbrooksite johnsenite-(Ce) senaite acetamide hydrohalite \
meridianiite chalcocyanite ekaterinite kamchatkite nitromagnesite rorisite scacchite sinjarite sveite tolbachite aplowite boothite \
chvaleticeite hohmannite hydrodresserite hydroscarbroite lonecreekite marthozite zaherite avogadrite carobbite chloraluminite cyanochroite \
ferruccite melanothallite palmierite piypite ponomarevite aubertite bayleyite caracolite fedotovite grimselite pseudograndreefite redingtonite wheatleyite wupatkite gregoryite \
natroxalate koktaite lecontite minasragrite ransomite \
ye'elimite chlorocalcite erythrosiderite gwihabaite molysite mikasaite tachyhydrite edoylerite metasideronatrite sideronatrite agaite andychristyite bairdite \
chromschieffelinite houseleyite markcooperite ottoite telluroperite hutcheonite allendeite fingerite mcbirneyite stoiberite ziesite"

client = MPDSDataRetrieval(dtype=MPDSDataTypes.ALL, api_key="KEY")
loss = []

for query in data.split(" "):
    try:
        answer_df = pd.DataFrame(client.get_data({"classes": query}))
    except:
        print(f"Not exist {query}")
        loss.append(query)

for i in loss:
    print(i)
print(len(loss))
