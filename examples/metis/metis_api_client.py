from metis_client import MetisAPI, MetisTokenAuth
from metis_client.dtos import MetisCalculationDTO

api_url = "http://localhost:3000"
test_engine = "pcrystal"
timeout_pcrystal = 360

file_path = "/home/alina/PycharmProjects/ml-playground/examples/data/ice_crystal"

try:
    with open(file_path, encoding="utf-8") as fp:
        CONTENT = fp.read()
except IndexError:
    print("Can't open file")

def on_progress_log(calc: MetisCalculationDTO):
    "Print progress"
    print("Progress:", calc.get("progress"))

client = MetisAPI(
    api_url,
    auth=MetisTokenAuth("admin@test.com"),
    timeout_pcrystal=timeout_pcrystal
)

print(client.v0.auth.whoami())
print("Engines are available:", client.calculations.supported())

data = client.v0.datasources.create(CONTENT)
calc = client.v0.calculations.create(data.get("id"), engine=test_engine)

results = client.v0.calculations.get_results(calc["id"], on_progress=on_progress_log)
print(results)
