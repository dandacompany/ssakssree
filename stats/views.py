from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from .utils import MixedConjointPriceRecommender
import pandas as pd
import io
import json

class ConjointView(APIView):
    def post(self, request):
        """
        POST 요청 처리:
        1. 요청 데이터에서 config와 데이터 추출
        2. MixedConjointPriceRecommender 객체 생성
        3. 모델 피팅
        4. 결과 분석 및 응답
        """
        try:
            # 디버깅을 위한 로그 추가
            print("Received data:", request.data)
            
            config = request.data.get('config')
            csv_data = request.data.get('data')
            
            # config가 문자열로 왔다면 JSON으로 파싱
            if isinstance(config, str):
                config = json.loads(config)
            
            if not config or not csv_data:
                return Response({"error": "config와 data 필수"}, status=400)
            
            # 데이터프레임 생성
            
            df = pd.read_csv(io.StringIO(csv_data))
            
            # 모델 생성 및 피팅
            
            recommender = MixedConjointPriceRecommender(config)
            recommender.fit_model(df)
            
            # 분석 결과 준비
            result = {
                "status": "success",
                "model_summary": str(recommender.model_result.summary()),
                "fixed_params": recommender.fixed_params,
                "dummy_map": recommender.dummy_map,
                "formula": recommender.formula_str
            }
            
            # 추가 분석 요청이 있으면 수행
            if 'analysis_request' in request.data:
                analysis_type = request.data['analysis_request'].get('type')
                
                if analysis_type == 'price_range':
                    # 특정 속성 조합에 대한 가격 범위 분석
                    attributes = request.data['analysis_request']['attributes']
                    price_range = recommender.find_price_range(
                        attributes=attributes,
                        price_min=request.data['analysis_request'].get('price_min', 5000),
                        price_max=request.data['analysis_request'].get('price_max', 30000),
                        step=request.data['analysis_request'].get('step', 1000),
                        threshold=request.data['analysis_request'].get('threshold', 4.5)
                    )
                    result['price_range'] = price_range
                
                elif analysis_type == 'feasible_combinations':
                    # 가능한 모든 조합 분석
                    feasible_combos = recommender.list_feasible_combinations(
                        price_min=request.data['analysis_request'].get('price_min', 5000),
                        price_max=request.data['analysis_request'].get('price_max', 30000),
                        step=request.data['analysis_request'].get('step', 1000),
                        threshold=request.data['analysis_request'].get('threshold', 4.5)
                    )
                    result['feasible_combinations'] = feasible_combos
            
            return Response(result)
            
        except Exception as e:
            # 더 자세한 에러 메시지
            import traceback
            print("Error:", str(e))
            print("Traceback:", traceback.format_exc())
            return Response({
                "status": "error",
                "message": str(e),
                "traceback": traceback.format_exc()
            }, status=500)
