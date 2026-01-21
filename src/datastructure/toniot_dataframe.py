import pandas as pd

class ToniotDataFrame(pd.DataFrame):
    """
    DataFrame spécifique au projet TON_IoT, permettant d'ajouter des méthodes métier et de garantir la cohérence des données.
    """
    @property
    def _constructor(self):
        return ToniotDataFrame

    def get_attack_traffic(self):
        """Retourne les lignes correspondant à du trafic d'attaque."""
        if 'label_str' in self.columns:
            return self[self['label_str'] == 'ddos']
        raise AttributeError("Colonne 'label_str' absente.")
