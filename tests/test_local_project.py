import os, secrets, tempfile, unittest, json
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient
from api_round2 import app
from backend.admin_auth import hash_password
from backend.local_predictor import LocalPredictor
from map_round2 import prediction_to_geojson, load_place_coordinates

class PublicReleaseTests(unittest.TestCase):
    def setUp(self): self.client=TestClient(app)
    def test_library_and_health(self):
        self.assertEqual(self.client.get('/health').status_code,200)
        self.assertEqual(self.client.get('/api/hexi-poetry/filters').json()['total_poems'],1230)
        self.assertEqual(self.client.get('/api/hexi-poetry/poems?limit=100').json()['total'],1230)
    def test_all_corpus_predictions_and_maps(self):
        predictor=LocalPredictor(); self.assertEqual(len(predictor.rows),1230)
        coords=load_place_coordinates()
        for row in predictor.rows:
            result=predictor.predict(row['text'])
            self.assertEqual(result['model']['mode'],'corpus_inference_replay')
            self.assertEqual(prediction_to_geojson(result,coords)['type'],'FeatureCollection')
    def test_semantic_map_api(self):
        text=LocalPredictor().rows[0]['text']
        for endpoint in ('/api/predict','/api/predict/map'):
            response=self.client.post(endpoint,json={'text':text})
            self.assertEqual(response.status_code,200,response.text)
    def test_unknown_is_explicit(self):
        result=LocalPredictor().predict('完全未知的文本')
        self.assertEqual(result['model']['mode'],'rules_unclassified')
        self.assertEqual(result['place']['confidence'],0)
    def test_visitor_denied_and_admin_disabled(self):
        self.assertFalse(self.client.get('/api/auth/session').json()['is_admin'])
        response=self.client.post('/api/poetry-image/nano-banana',json={'text':'test'})
        self.assertEqual(response.status_code,403)
        with patch.dict(os.environ,{'ADMIN_PASSWORD_HASH':'','ADMIN_SESSION_SECRET':''}):
            self.assertEqual(self.client.post('/api/auth/login',json={'password':'unset'}).status_code,503)
    def test_admin_login_csrf_disabled_billing_logout(self):
        password=secrets.token_urlsafe(24)
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ,{'ADMIN_PASSWORD_HASH':hash_password(password),'ADMIN_SESSION_SECRET':secrets.token_urlsafe(48),'ADMIN_SESSION_DB':str(Path(tmp)/'sessions.db'),'AUTH_COOKIE_SECURE':'false','ENABLE_PAID_APIS':'false'}):
            c=TestClient(app)
            response=c.post('/api/auth/login',json={'password':password})
            self.assertEqual(response.status_code,200,response.text)
            csrf=response.json()['csrf_token']
            self.assertTrue(c.get('/api/auth/session').json()['is_admin'])
            response=c.post('/api/poetry-image/nano-banana',json={'text':'test'},headers={'X-CSRF-Token':csrf})
            self.assertEqual(response.status_code,503,response.text)
            self.assertEqual(c.post('/api/auth/logout').status_code,403)
            self.assertEqual(c.post('/api/auth/logout',headers={'X-CSRF-Token':csrf}).status_code,200)
            self.assertFalse(c.get('/api/auth/session').json()['is_admin'])
    def test_reality_upstream_failure_is_honest(self):
        import requests
        with patch('requests.get',side_effect=requests.RequestException('offline')):
            r=self.client.post('/api/reality/search',json={'text':'玉门关','location':'玉门关'})
        self.assertEqual(r.status_code,200,r.text)
        self.assertFalse(r.json()['ok'])
        self.assertFalse(r.json()['billing']['paid_api'])

    def test_local_demos_complete_and_composable(self):
        cases=self.client.get('/api/demo/cases').json()
        self.assertEqual(len(cases),3)
        for case in cases:
            self.assertTrue(all(case['availability'].values()),case['id'])
            response=self.client.get('/api/demo/cases/'+case['id'])
            self.assertEqual(response.status_code,200,response.text)
            result=response.json()['result']
            self.assertTrue(result['map']['features'])
            self.assertTrue(result['image']['image_url'].startswith('/static/demo/'))
            self.assertTrue(result['ai_video']['video_url'].startswith('/static/demo/'))
            response=self.client.post('/api/main-stage/compose',json={'text':case['content']})
            self.assertEqual(response.status_code,200,response.text)
    def test_non_demo_composition_without_network(self):
        fake={'ok':False,'status':'not_requested','query':'test','best':None,'candidates':[],'message':'offline','warnings':[]}
        with patch('backend.service.video_search.reality.search_reality',return_value=fake):
            response=self.client.post('/api/main-stage/compose',json={'text':'未收入诗库的诗句，玉门关'})
        self.assertEqual(response.status_code,200,response.text)
        self.assertIn('storyContext',response.json())

if __name__=='__main__': unittest.main()
