from django.test import TestCase, Client
from rest_framework.test import APITestCase
from django.urls import reverse
import pandas as pd
import io

class ConjointAnalysisTests(APITestCase):
    def setUp(self):
        self.client = Client()
        self.url = reverse('conjoint')
        
        # 샘플 데이터 로드
        with open('stats/conjoint_sample_data_extended.csv', 'r') as f:
            self.sample_data = f.read()
            
        # 테스트용 config 설정 (utils.py 참조)
        self.config = {
            'respondent_col': 'RespondentID',
            'rating_col': 'Rating',
            'price_col': 'Price',
            'categorical_cols': ['Brand', 'SolutionQuality', 'Difficulty', 'Design'],
            'interaction_cols': ['Brand', 'Design'],
            'price_scale_factor': 1000,
            'baseline_levels': {
                'Brand': 'Local',
                'SolutionQuality': 'Basic',
                'Difficulty': 'Easy', 
                'Design': '단색'
            },
            'threshold': 4.5  # utils.py에서 사용하는 threshold 추가
        }

    def test_conjoint_analysis(self):
        # POST 요청 데이터 준비
        data = {
            'config': self.config,
            'data': self.sample_data
        }
        
        # API 요청
        response = self.client.post(self.url, data, format='json')
        
        # 응답 검증
        self.assertEqual(response.status_code, 200)
        self.assertIn('model_summary', response.data)
        self.assertIn('fixed_params', response.data)
        self.assertIn('dummy_map', response.data)
        self.assertIn('formula', response.data)
        
    def test_price_range_analysis(self):
        # 추가 분석 테스트 (conjoint_sample_data_extended.csv 컬럼명에 맞게 수정)
        data = {
            'config': self.config,
            'data': self.sample_data,
            'analysis_request': {
                'type': 'price_range',
                'attributes': {
                    'Brand': 'National',
                    'SolutionQuality': 'Advanced',
                    'Difficulty': 'Medium',
                    'Design': '풀컬러'
                },
                'price_min': 10000,
                'price_max': 30000,
                'step': 1000,
                'threshold': 4.5
            }
        }
        
        response = self.client.post(self.url, data, format='json')
        self.assertEqual(response.status_code, 200)
        self.assertIn('price_range', response.data)
        
    def test_feasible_combinations(self):
        # 가능한 조합 분석 테스트
        data = {
            'config': self.config,
            'data': self.sample_data,
            'analysis_request': {
                'type': 'feasible_combinations',
                'price_min': 10000,
                'price_max': 30000,
                'step': 1000,
                'threshold': 4.5
            }
        }
        
        response = self.client.post(self.url, data, format='json')
        self.assertEqual(response.status_code, 200)
        self.assertIn('feasible_combinations', response.data)
