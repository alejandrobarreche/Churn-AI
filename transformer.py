from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
import pandas as pd

class BinaryEncoder(BaseEstimator, TransformerMixin):


    binary_sinno_columns = [
        'SEGURO_BATERIA_LARGO_PLAZO', 'EN_GARANTIA',
    ]

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()

        for col in self.binary_sinno_columns:
            X[col] = X[col].map({'SI': 1, 'NO': 0})


        X['MANTENIMIENTO_GRATUITO'] *= 0.25
        X['QUEJA'] = X['QUEJA'].fillna('NO').map({'SI': 1, 'NO': 0})
        X['Churn_400'] = X['Churn_400'].map({'Y': 1, 'N': 0})

        return X


class FrequencyEncoder(BaseEstimator, TransformerMixin):

    high_cardinality_columns = ['Modelo', 'PROV_DESC']

    def __init__(self):
        self.frequency_maps_ = {}

    def fit(self, X, y=None):
        for col in self.high_cardinality_columns:
            self.frequency_maps_[col] = X[col].value_counts(normalize=True)
        return self

    def transform(self, X, y=None):
        X = X.copy()
        for col in self.high_cardinality_columns:
            X[col] = X[col].map(self.frequency_maps_[col]).fillna(0)
        return X


class OrdinalExtensionEncoder(BaseEstimator, TransformerMixin):
    """
    OrdinalEncoder con orden explícito para EXTENSION_GARANTIA.
    'SI, Campa a regalo' se trata como 'SI, Financiera'.
    """
    REMAP = {'SI, Campa a Regalo': 'SI, Financiera'}
    ORDER = ['NO', 'SI', 'SI, Financiera']

    def __init__(self):
        # 1. Le indicamos el orden explícito que queremos que respete
        self.encoder = OrdinalEncoder(categories=[self.ORDER])

    def fit(self, X, y=None):
        Xt = X[['EXTENSION_GARANTIA']].copy()
        # Remapeamos antes de ajustar
        Xt['EXTENSION_GARANTIA'] = Xt['EXTENSION_GARANTIA'].replace(self.REMAP)
        self.encoder.fit(Xt)
        return self

    def transform(self, X, y=None):
        X = X.copy()
        # 2. PRIMERO REMAPEAMOS en los datos nuevos (fundamental para que el encoder lo entienda)
        X['EXTENSION_GARANTIA'] = X['EXTENSION_GARANTIA'].replace(self.REMAP)

        # 3. LUEGO HACEMOS EL ENCODING
        X['EXTENSION_GARANTIA'] = self.encoder.transform(X[['EXTENSION_GARANTIA']])

        return X


class OrdinalEquipamientoEncoder(BaseEstimator, TransformerMixin):

    ORDER = ['Low', 'Mid', 'Mid-High', 'High']

    def __init__(self):
        self.encoder = OrdinalEncoder(
            categories=[self.ORDER],
            handle_unknown='use_encoded_value',
            unknown_value=-1,
        )

    def fit(self, X, y=None):
        self.encoder.fit(X[['Equipamiento']].astype(str))
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X['Equipamiento'] = self.encoder.transform(X[['Equipamiento']].astype(str))
        return X


class NominalOneHotEncoder(BaseEstimator, TransformerMixin):
    """
    OneHotEncoder para todas las variables nominales sin orden implícito,
    incluyendo las de media/alta cardinalidad no cubiertas por FrequencyEncoder.
    """

    nominal_columns = [
        # categorical_labeled_variables nominales
        'MOTIVO_VENTA', 'GENERO', 'Fuel', 'TRANSMISION_ID', 'Origen',
        # categorical_multilabeled_variables nominales
        'FORMA_PAGO', 'STATUS_SOCIAL', 'TIPO_CARROCERIA', 'ZONA',
    ]

    def __init__(self):
        self.encoder = OneHotEncoder(
            sparse_output=False,
            handle_unknown='ignore',
        )
        self.feature_names_ = None

    def fit(self, X, y=None):
        self.encoder.fit(X[self.nominal_columns])
        self.feature_names_ = self.encoder.get_feature_names_out(self.nominal_columns)
        return self

    def transform(self, X, y=None):
        X = X.copy()
        encoded = self.encoder.transform(X[self.nominal_columns])
        encoded_df = pd.DataFrame(encoded, columns=self.feature_names_, index=X.index)
        X.drop(columns=self.nominal_columns, inplace=True)
        return pd.concat([X, encoded_df], axis=1)


class ColumnDropper(BaseEstimator, TransformerMixin):
    """
    Elimina columnas sin poder predictivo.
    Cambia unuseful_columns para experimentar con distintas combinaciones.
    """

    unuseful_columns = [
        # Identificadores
        'CODE', 'Id_Producto', 'Customer_ID',
        # Fechas brutas
        'Sales_Date', 'FIN_GARANTIA', 'BASE_DATE',
        # Leakage económico
        'Margen_eur_bruto', 'Margen_eur', 'COSTE_VENTA_NO_IMPUESTOS',
        # Geografía redundante
        'CODIGO_POSTAL', 'ZONA',
        # Taller redundante 'Revisiones',
        'TIENDA_DESC', 'Km_medio_por_revision', 'km_ultima_revision',
        # Sensibles
        'STATUS_SOCIAL', 'GENERO',
        # Resto
        'ENCUESTA_CLIENTE_ZONA_TALLER', 'DAYS_LAST_SERVICE', 'Revisiones'

    ]

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        # Las columnas OneHot generadas llevan prefijo (ej: GENERO_H, ZONA_Norte)
        # por eso buscamos también por prefijo
        cols_to_drop = [
            c for c in X.columns
            if c in self.unuseful_columns
               or any(c.startswith(f'{base}_') for base in self.unuseful_columns)
        ]
        X.drop(columns=cols_to_drop, inplace=True, errors='ignore')
        return X



class GastoRelativoEncoder(BaseEstimator, TransformerMixin):
    """
    Crea la feature 'gasto_relativo' = PVP / RENTA_MEDIA_ESTIMADA.
    Mide el esfuerzo económico relativo del cliente para comprar el vehículo.
    Debe ejecutarse ANTES de PriceStandard (PVP aún en euros).
    clip(lower=1) evita división por cero antes de que InstanceDropper actúe.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X['gasto_relativo'] = X['PVP'] / X['RENTA_MEDIA_ESTIMADA'].clip(lower=1)
        return X


class PriceStandard(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X['PVP'] = X['PVP'] / 1000
        return X



class InstanceDropper(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        # Elimina las instancias con RENTA_MEDIA_ESTIMADA = 0
        X = X[X['RENTA_MEDIA_ESTIMADA'] != 0]
        return X