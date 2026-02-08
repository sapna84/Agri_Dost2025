from backend import app as backend_app

client = backend_app.app.test_client()
print('Requesting /status')
r = client.get('/status')
print('status', r.status_code)
print(r.get_json())
