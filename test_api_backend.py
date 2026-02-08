from backend import app as backend_app

client = backend_app.app.test_client()
print('Requesting /api/market-prices')
r = client.get('/api/market-prices')
print('status', r.status_code)
print(r.get_json())

print('\nRequesting /api/market-prices?crop=Wheat')
r2 = client.get('/api/market-prices?crop=Wheat')
print('status', r2.status_code)
print(r2.get_json())
