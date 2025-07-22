"""
Econometric tests for Rational Expectations Hypothesis analysis.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats
from typing import Dict, Optional, Tuple
import logging

# Configure logging
logger = logging.getLogger(__name__)


class REHTests:
    """
    Collection of econometric tests for Rational Expectations Hypothesis analysis
    """
    
    @staticmethod
    def mincer_zarnowitz_test(forecast: pd.Series, realized: pd.Series) -> Dict:
        """
        Perform Mincer-Zarnowitz test for forecast unbiasedness and efficiency
        
        Regression: P_t = α + β · E_{t-h}[P_t] + u_t
        H_0: (α, β) = (0, 1)
        
        Parameters:
        - forecast: Series of forecasts
        - realized: Series of realized values
        
        Returns:
        - Dict: Test results including coefficients, p-values, and joint test
        """
        logger.info("Running Mincer-Zarnowitz test...")
        
        try:
            # Ensure both series are aligned
            aligned_data = pd.DataFrame({'forecast': forecast, 'realized': realized}).dropna()
            
            if len(aligned_data) < 10:
                raise ValueError("Insufficient data for Mincer-Zarnowitz test")
            
            # Set up regression
            X = sm.add_constant(aligned_data['forecast'])
            y = aligned_data['realized']
            
            # Fit model
            model = sm.OLS(y, X).fit()
            
            # Joint test: (α, β) = (0, 1)
            restrictions = "const = 0, forecast = 1"
            joint_test = model.f_test(restrictions)
            
            results = {
                "alpha": float(model.params["const"]),
                "beta": float(model.params["forecast"]),
                "alpha_pvalue": float(model.pvalues["const"]),
                "beta_pvalue": float(model.pvalues["forecast"]),
                "alpha_stderr": float(model.bse["const"]),
                "beta_stderr": float(model.bse["forecast"]),
                "joint_test_fstat": float(joint_test.fvalue[0][0]),
                "joint_test_pvalue": float(joint_test.pvalue),
                "r_squared": float(model.rsquared),
                "adj_r_squared": float(model.rsquared_adj),
                "durbin_watson": float(sm.stats.stattools.durbin_watson(model.resid)),
                "n_observations": len(aligned_data),
                "passes_joint_test": float(joint_test.pvalue) > 0.05
            }
            
            logger.info(f"Mincer-Zarnowitz test completed: Joint p-value = {joint_test.pvalue:.4f}")
            return results
            
        except Exception as e:
            logger.error(f"Mincer-Zarnowitz test failed: {e}")
            return {"error": str(e)}
    
    @staticmethod
    def autocorrelation_test(forecast_errors: pd.Series, max_lags: int = 10) -> Dict:
        """
        Test forecast errors for autocorrelation using Ljung-Box test
        
        Parameters:
        - forecast_errors: Series of forecast errors
        - max_lags: Maximum number of lags to test
        
        Returns:
        - Dict: Test results including statistics and p-values
        """
        logger.info("Running autocorrelation tests...")
        
        try:
            errors_clean = forecast_errors.dropna()
            
            if len(errors_clean) < max_lags * 2:
                max_lags = max(1, len(errors_clean) // 4)
                logger.warning(f"Reduced max_lags to {max_lags} due to sample size")
            
            if max_lags < 1:
                return {"error": "Insufficient data for autocorrelation test"}
            
            # Ljung-Box test
            lb_test = acorr_ljungbox(errors_clean, lags=max_lags, return_df=True)
            
            # Extract results
            results = {
                "ljung_box_stat": float(lb_test["lb_stat"].iloc[-1]) if not lb_test.empty else np.nan,
                "ljung_box_pvalue": float(lb_test["lb_pvalue"].iloc[-1]) if not lb_test.empty else np.nan,
                "significant_autocorr": bool((lb_test["lb_pvalue"] < 0.05).any()) if not lb_test.empty else False,
                "max_lags_tested": int(max_lags),
                "n_observations": len(errors_clean),
                "passes_efficiency_test": not bool((lb_test["lb_pvalue"] < 0.05).any()) if not lb_test.empty else False
            }
            
            # Individual lag results
            results["lag_results"] = {
                f"lag_{i+1}": {
                    "lb_stat": float(lb_test["lb_stat"].iloc[i]),
                    "lb_pvalue": float(lb_test["lb_pvalue"].iloc[i])
                }
                for i in range(min(len(lb_test), 5))  # Store first 5 lags
            } if not lb_test.empty else {}
            
            logger.info(f"Autocorrelation test completed: Significant autocorr = {results['significant_autocorr']}")
            return results
            
        except Exception as e:
            logger.error(f"Autocorrelation test failed: {e}")
            return {"error": str(e)}
    
    @staticmethod
    def bias_test(forecast_errors: pd.Series) -> Dict:
        """
        Test forecast errors for systematic bias (Holden-Peel test)
        
        H_0: Mean forecast error = 0
        
        Parameters:
        - forecast_errors: Series of forecast errors
        
        Returns:
        - Dict: Test results including mean error, t-statistic, p-value
        """
        logger.info("Running bias test...")
        
        try:
            errors_clean = forecast_errors.dropna()
            
            if len(errors_clean) < 5:
                return {"error": "Insufficient data for bias test"}
            
            # Test if mean error is significantly different from zero
            mean_error = errors_clean.mean()
            std_error = errors_clean.std()
            n = len(errors_clean)
            
            # t-test for zero mean
            t_stat, p_value = stats.ttest_1samp(errors_clean, 0)
            
            # Additional statistics
            confidence_interval = stats.t.interval(
                0.95, n-1, loc=mean_error, scale=std_error/np.sqrt(n)
            )
            
            results = {
                "mean_error": float(mean_error),
                "std_error": float(std_error),
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "confidence_interval_95": [float(confidence_interval[0]), float(confidence_interval[1])],
                "is_biased": bool(p_value < 0.05),
                "bias_direction": "overestimation" if mean_error < 0 else "underestimation",
                "n_observations": int(n),
                "passes_unbiasedness_test": bool(p_value >= 0.05)
            }
            
            logger.info(f"Bias test completed: Mean error = {mean_error:.4f}, p-value = {p_value:.4f}")
            return results
            
        except Exception as e:
            logger.error(f"Bias test failed: {e}")
            return {"error": str(e)}
    
    @staticmethod
    def orthogonality_test(
        forecast_errors: pd.Series, 
        external_vars: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        Test forecast errors for orthogonality with respect to available information
        
        Regression: e_t = α + Σβᵢ Xᵢ,t-h + u_t
        H_0: β₁ = β₂ = ... = βₖ = 0
        
        Parameters:
        - forecast_errors: Series of forecast errors
        - external_vars: DataFrame of external variables available at forecast time
        
        Returns:
        - Dict: Test results including F-statistic and p-value
        """
        if external_vars is None or external_vars.empty:
            logger.warning("No external variables provided for orthogonality test")
            return {"error": "No external variables provided"}
        
        logger.info("Running orthogonality test...")
        
        try:
            # Align data
            combined_data = pd.concat([forecast_errors, external_vars], axis=1, join='inner')
            combined_data = combined_data.dropna()
            
            if len(combined_data) < 10:
                return {"error": "Insufficient aligned data for orthogonality test"}
            
            # Prepare regression
            y = combined_data.iloc[:, 0]  # forecast errors
            X = combined_data.iloc[:, 1:]  # external variables
            X = sm.add_constant(X)
            
            # Fit model
            model = sm.OLS(y, X).fit()
            
            # F-test for joint significance of external variables (excluding constant)
            var_names = [col for col in X.columns if col != 'const']
            if not var_names:
                return {"error": "No valid external variables for testing"}
            
            # Joint significance test
            f_test = model.f_test(f"[{' = 0, '.join(var_names)} = 0]")
            
            results = {
                "f_statistic": float(f_test.fvalue[0][0]) if f_test.fvalue.size > 0 else np.nan,
                "f_pvalue": float(f_test.pvalue) if hasattr(f_test, 'pvalue') else np.nan,
                "r_squared": float(model.rsquared),
                "n_observations": len(combined_data),
                "n_variables": len(var_names),
                "passes_orthogonality_test": bool(f_test.pvalue >= 0.05) if hasattr(f_test, 'pvalue') else False
            }
            
            # Individual variable results
            results["variable_results"] = {}
            for var in var_names:
                if var in model.params.index:
                    results["variable_results"][var] = {
                        "coefficient": float(model.params[var]),
                        "pvalue": float(model.pvalues[var]),
                        "stderr": float(model.bse[var])
                    }
            
            logger.info(f"Orthogonality test completed: F p-value = {f_test.pvalue:.4f}")
            return results
            
        except Exception as e:
            logger.error(f"Orthogonality test failed: {e}")
            return {"error": str(e)}
    
    @staticmethod
    def comprehensive_reh_assessment(
        forecast: pd.Series,
        realized: pd.Series,
        external_vars: Optional[pd.DataFrame] = None,
        max_autocorr_lags: int = 10
    ) -> Dict:
        """
        Run complete suite of REH tests and provide overall assessment
        
        Parameters:
        - forecast: Series of forecasts
        - realized: Series of realized values  
        - external_vars: Optional external variables for orthogonality test
        - max_autocorr_lags: Maximum lags for autocorrelation test
        
        Returns:
        - Dict: Comprehensive test results and rationality assessment
        """
        logger.info("Running comprehensive REH assessment...")
        
        # Calculate forecast errors
        aligned_data = pd.DataFrame({'forecast': forecast, 'realized': realized}).dropna()
        forecast_errors = aligned_data['realized'] - aligned_data['forecast']
        
        results = {}
        
        # Run individual tests
        results["mincer_zarnowitz"] = REHTests.mincer_zarnowitz_test(
            aligned_data['forecast'], aligned_data['realized']
        )
        
        results["autocorrelation"] = REHTests.autocorrelation_test(
            forecast_errors, max_lags=max_autocorr_lags
        )
        
        results["bias_test"] = REHTests.bias_test(forecast_errors)
        
        if external_vars is not None and not external_vars.empty:
            results["orthogonality"] = REHTests.orthogonality_test(
                forecast_errors, external_vars
            )
        
        # Overall rationality assessment
        try:
            mz_rational = results.get("mincer_zarnowitz", {}).get("passes_joint_test", False)
            efficient = results.get("autocorrelation", {}).get("passes_efficiency_test", False)  
            unbiased = results.get("bias_test", {}).get("passes_unbiasedness_test", False)
            orthogonal = results.get("orthogonality", {}).get("passes_orthogonality_test", True)  # Default True if no test
            
            results["rationality_assessment"] = {
                "unbiased": unbiased,
                "mz_rational": mz_rational, 
                "efficient": efficient,
                "orthogonal": orthogonal,
                "overall_rational": unbiased and mz_rational and efficient and orthogonal,
                "assessment_date": pd.Timestamp.now().isoformat(),
                "n_observations": len(aligned_data)
            }
            
        except Exception as e:
            logger.error(f"Rationality assessment failed: {e}")
            results["rationality_assessment"] = {"error": str(e)}
        
        logger.info("Comprehensive REH assessment completed")
        return results