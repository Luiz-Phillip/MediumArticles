from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
import pandas as pd
def vif(df: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
  """
  Função que calcula o VIF para cada uma das variáveis e retorna uma Tabela com
  os valores ordenados de forma descrescente.
  Args:
    df (pd.core.frame.DataFrame) - DataFrame com o conjunto das variáveis de entradas
  Returns:
    result (pd.core.frame.DataFrame) - DataFrame com as variáveis no Index e com a Coluna VIF.
  """
  n_vars = df.shape[1]
  x = StandardScaler().fit_transform(df)
  vif_value = [variance_inflation_factor(x, i) for i in range(n_vars)]
  result = pd.DataFrame(vif_value, index = df.columns, columns= ['VIF'])
  return result.sort_values(by = 'VIF', ascending=False)
