import pandas as pd
import statsmodels.formula.api as smf
import numpy as np
import itertools

class MixedConjointPriceRecommender:
    def __init__(self, config):
        """
        개선 사항:
         - price_scale_factor: 가격 스케일 조정 (default=1), ex) 1000 => Price를 천원 단위로 전환
         - baseline_levels: {col: level}, baseline으로 삼고 싶은 수준 지정 (optional)
         
        예시 config:
        {
            'respondent_col': 'RespondentID',
            'rating_col': 'Rating',
            'price_col': 'Price',
            'categorical_cols': ['Brand','SolutionQuality','Difficulty','Design'],
            'interaction_cols': ['Brand','Design'],
            'price_scale_factor': 1000,
            'baseline_levels': {
                'Brand': 'Local',
                'SolutionQuality': 'Basic',
                'Difficulty': 'Easy',
                'Design': '단색'
            }
        }
        """
        self.config = config
        self.model_result = None
        self.fixed_params = None
        
        self.dummy_map = {}    # {col -> {level -> dummy_col명 or None(baseline)}}
        self.formula_str = None
        self.df_dummies = None

        # 내부적으로 scaled price 컬럼명을 정해줌
        self.price_col = config['price_col']
        self.price_scaled_col = self.price_col + "_scaled"
        
    def _reorder_category_for_baseline(self, df, col, baseline_level):
        """
        (선택적) baseline_level이 지정되어 있으면,
        df[col]의 카테고리 순서를 baseline_level이 첫 번째가 되도록 재배치
        """
        if baseline_level is None:
            return df  # 아무것도 안 함
        
        # col의 기존 값
        if baseline_level not in df[col].unique():
            # 지정한 baseline_level이 실제 데이터에 없다면 무시
            return df
        
        # pd.Categorical로 순서 재설정
        unique_levels = list(df[col].unique())
        # baseline_level을 맨 앞으로
        new_cat_order = [baseline_level] + [lvl for lvl in unique_levels if lvl != baseline_level]
        df[col] = pd.Categorical(df[col], categories=new_cat_order, ordered=False)
        return df
    
    def fit_model(self, df):
        """
        1) 데이터 로딩
        2) baseline_levels 적용 -> 범주 순서 재배치
        3) Price 스케일링
        4) 범주형 -> get_dummies
        5) 상호작용 항
        6) MixedLM 적합
        """
        
        # (1) 필요한 변수 추출
        respondent_col = self.config['respondent_col']
        rating_col = self.config['rating_col']
        cat_cols = self.config['categorical_cols']
        interaction_cols = self.config.get('interaction_cols', [])
        
        # (2) baseline_levels 적용
        baseline_levels = self.config.get('baseline_levels', {})
        for col in cat_cols:
            bl_level = baseline_levels.get(col, None)
            df = self._reorder_category_for_baseline(df, col, bl_level)
        
        # (3) Price 스케일링
        price_scale_factor = self.config.get('price_scale_factor', 1.0)
        # 새 컬럼: Price_scaled
        df[self.price_scaled_col] = df[self.price_col] / price_scale_factor
        
        # (4) get_dummies
        df_dummies = pd.get_dummies(df, columns=cat_cols, drop_first=True)
        
        # (5) dummy_map 구성
        for col in cat_cols:
            unique_levels = df[col].dropna().unique()
            # drop_first=True로 인해 첫 번째 수준이 baseline
            # (reorder_category_for_baseline에서 baseline이 맨 앞)
            baseline_level = unique_levels[0] if len(unique_levels)>1 else None
            
            self.dummy_map[col] = {}
            for lvl in unique_levels:
                if lvl == baseline_level:
                    self.dummy_map[col][lvl] = None
                else:
                    # 실제 get_dummies로 생성된 컬럼명
                    dummy_col = f"{col}_{lvl}"
                    if dummy_col in df_dummies.columns:
                        self.dummy_map[col][lvl] = dummy_col
                    else:
                        self.dummy_map[col][lvl] = None
        
        # (6) 상호작용 항 생성
        # Price_scaled × 각 dummy
        for col in interaction_cols:
            mapping = self.dummy_map[col]
            for lvl, dummy_col in mapping.items():
                if dummy_col:  # baseline이면 None
                    new_col = f"{self.price_scaled_col}_x_{dummy_col}"
                    df_dummies[new_col] = df_dummies[self.price_scaled_col] * df_dummies[dummy_col]
        
        self.df_dummies = df_dummies
        
        # (7) formula 생성
        main_terms = [self.price_scaled_col]  # scaled price 주효과
        for c in df_dummies.columns:
            if c not in [respondent_col, rating_col, self.price_col, self.price_scaled_col]:
                main_terms.append(c)
        
        self.formula_str = f"{rating_col} ~ " + " + ".join(main_terms)
        
        # (8) MixedLM (Random Intercept)
        mixed_model = smf.mixedlm(self.formula_str, df_dummies, groups=df_dummies[respondent_col])
        result = mixed_model.fit(method='lbfgs')
        
        self.model_result = result
        self.fixed_params = dict(result.params)
        
        print("=== Enhanced Mixed Model Fitting Complete ===")
        print("Price Scale Factor =", price_scale_factor)
        print("Formula:", self.formula_str)
        print(result.summary())
    
    def predict_mean_rating(self, price, attributes: dict):
        """
        평균 응답자(랜덤효과=0) 가정 하에, 주어진 price(실제원)와 속성 조합(attributes)을 넣었을 때의 평점 예측
        -> 내부적으로 price_scaled = price / price_scale_factor
        -> baseline/dummy + 상호작용 고려
        attributes 예: {'Brand':'National', 'SolutionQuality':'Advanced', ...}
        """
        if self.fixed_params is None:
            raise RuntimeError("Model not fitted. Call fit_model() first.")
        
        # 1) price_scaled
        price_scale_factor = self.config.get('price_scale_factor', 1.0)
        price_scaled = price / price_scale_factor
        
        rating_pred = self.fixed_params.get('Intercept', 0.0)
        
        # 2) price_scaled 주효과
        if self.price_scaled_col in self.fixed_params:
            rating_pred += self.fixed_params[self.price_scaled_col] * price_scaled
        
        # 3) 범주형 dummy
        for col, lvl in attributes.items():
            if col in self.dummy_map:
                dummy_col = self.dummy_map[col].get(lvl, None)
                if dummy_col and dummy_col in self.fixed_params:
                    rating_pred += self.fixed_params[dummy_col]
        
        # 4) 상호작용
        for col, lvl in attributes.items():
            if col in self.dummy_map:
                dummy_col = self.dummy_map[col].get(lvl, None)
                if dummy_col:
                    inter_col = f"{self.price_scaled_col}_x_{dummy_col}"
                    if inter_col in self.fixed_params:
                        rating_pred += self.fixed_params[inter_col] * price_scaled
        
        return rating_pred
    
    def find_price_range(self, attributes: dict, price_min=5000, price_max=30000, step=1000, threshold=4.5):
        """
        속성 + 가격 범위 스캔 -> threshold 이상 되는 구간 반환
        """
        valid_prices = []
        for p in range(price_min, price_max+1, step):
            r = self.predict_mean_rating(p, attributes)
            if r >= threshold:
                valid_prices.append(p)
        
        if not valid_prices:
            return None
        return (min(valid_prices), max(valid_prices))
    
    def list_feasible_combinations(self, price_min=5000, price_max=30000, step=1000, threshold=4.5):
        """
        모든 범주형 조합에 대해 find_price_range -> feasible한 조합만 리턴
        """
        cat_cols = self.config['categorical_cols']
        
        # (a) 각 컬럼별 수준 수집
        levels_for_col = {}
        for col in cat_cols:
            # dummy_map[col] = { level -> dummy_col or None (baseline) }
            # => key만 수집
            all_levels = list(self.dummy_map[col].keys())
            levels_for_col[col] = all_levels
        
        # (b) 모든 조합
        all_combinations = []
        col_order = list(cat_cols)
        list_of_levels = [levels_for_col[c] for c in col_order]
        
        for combo in itertools.product(*list_of_levels):
            attr_dict = dict(zip(col_order, combo))
            all_combinations.append(attr_dict)
        
        # (c) 각 조합에 대해 price_range 검색
        feasible_results = []
        for attr_dict in all_combinations:
            prange = self.find_price_range(
                attributes=attr_dict,
                price_min=price_min,
                price_max=price_max,
                step=step,
                threshold=threshold
            )
            if prange is not None:
                feasible_results.append((attr_dict, prange))
        
        return feasible_results

def example():
    # 1) config 예시
    config = {
        'respondent_col': 'RespondentID',
        'rating_col': 'Rating',
        'price_col': 'Price',
        'categorical_cols': ['Brand','SolutionQuality','Difficulty','Design'],
        'interaction_cols': ['Brand','Design'],

        # 개선 사항:
        'price_scale_factor': 1000,  # Price를 천원 단위로 스케일링
        'baseline_levels': {
            # 예시: "Local"을 baseline으로 삼고 싶다
            'Brand': 'Local',
            'SolutionQuality': 'Basic',
            'Difficulty': 'Easy',
            'Design': '단색'
        },
        'threshold': 3
    }

    # 2) 모델 객체 생성 & 피팅
    data_path = "conjoint_sample_data_extended.csv"  # 실제 파일 경로
    df = pd.read_csv(data_path)
    recommender = MixedConjointPriceRecommender(config)
    recommender.fit_model(df)

    # 3) 특정 조합 & 임계값 테스트
    new_attrs = {
        'Brand': 'National', 
        'SolutionQuality': 'Advanced',
        'Difficulty': 'Easy', 
        'Design': '단색'
    }
    price_range = recommender.find_price_range(new_attrs, threshold=config['threshold'])
    print("\n--- Single Combination Test ---")
    print("Attributes =", new_attrs)
    print(f"Price Range >= {config['threshold']} rating:", price_range)

    # 4) 전수조합 탐색
    feasible_combos = recommender.list_feasible_combinations(price_min=5000, price_max=30000, step=1000, threshold=config['threshold'])
    print(f"\n=== Found {len(feasible_combos)} feasible combos for >={config['threshold']} ===")
    for (attrs, pr) in feasible_combos[:10]:
        print("Attrs:", attrs, " =>", pr)