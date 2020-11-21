from typing import Dict, Tuple, Text
import pandas as pd
import numpy as np


def min2(current: Tuple[Text, float], new: Tuple[Text, float]) -> Text:
    """ Function for calculating updated minimum """
    val = current[0]
    if current[1] > new[1]:
        val = new[0]

    return val


class fixed_income_filter:
    """ Based Income Filter Object """

    def __init__(self, options: Dict, df, dbg=False):
        """ Constructor for base Fixed Income """
        self.options = options.copy()
        self.dbg = dbg

        if isinstance(df, pd.DataFrame) and all(df.shape) > 0:
            self.df = df.copy()
        else:
            raise ValueError("DF must be pandas dataframe, with non-zero shape")

        self.test_config()

    def test_config(self):
        ''' tests configuration '''
        if "key" in self.options.keys() and "value" in self.options.keys():
            if self.dbg and 'exclude' not in self.options.keys():
                print("Warn -- exclude is missing (defaults to True)")
            if  self.dbg and 'func' not in self.options.keys():
                print("Warn -- func is missing")
        else:
            if self.dbg:
                print(self.options)
            raise ValueError("fixed_income filter: missing config values")


    def include(self):
        """ produces series of booleans indicating inclusions """
        if 'exclude' in self.options.keys() and self.options['exclude'] and\
                self.options['exclude'] in self.df.columns:
            inc_ind = np.logical_not(self.df[self.options["exclude"]])
        else:
            inc_ind = pd.Series(np.ones(self.df.shape[0]) > 0, self.df.index)

        if "func" not in self.options.keys():
            raise ValueError("Missing function definition")

        if str(self.options["func"]).upper().endswith("NOT"):
            inc_ind = np.logical_not(inc_ind)
        elif str(self.options["func"]).upper().startswith("ENDSW"):
            ind = self.df[self.options["key"]].str.upper().str.endswith(
                self.options["value"].upper())
            inc_ind = np.logical_and(inc_ind, ind)
        elif str(self.options["func"]).upper().startswith("STARTSW"):
            ind = self.df[self.options["key"]].str.upper().str.startswith(
                self.options["value"].upper())
            inc_ind = np.logical_and(inc_ind, ind)

        else:
            raise ValueError("Faulty function definition")
        return inc_ind

class fixed_income_filter_logical():
    ''' Filter for logical (and, or xor) filters '''
    def __init__(self, options: Dict, df, dbg=False):
        """ Constructor for base Fixed Income """
        self.options = options.copy()
        self.dbg = dbg

        if isinstance(df, pd.DataFrame) and all(df.shape) > 0:
            self.df = df.copy()
        else:
            raise ValueError("DF must be pandas dataframe, with non-zero shape")

        self.test_config()
        self.tests = []
        for itm in ["first", "second"]:
            if "type" in self.options[itm].keys() and\
                    self.options[itm]["type"].upper() == "NUMERIC":
                self.tests.append(fixed_income_filter_numeric(self.options[itm], df, self.dbg))
            else:
                self.tests.append(fixed_income_filter(self.options[itm], df, self.dbg))

    def test_config(self):
        ''' tests configuration '''
        if len(self.options) > 1 and 'first' in self.options.keys() and\
                "second" in self.options.keys():
            if self.dbg and 'operator' not in self.options.keys():
                print("Warn -- operator is missing (defaults to determination name")
        else:
            raise ValueError("fixed_income filter: missing config values")

    def include(self):
        ''' applies operator to two valid tests '''

        if "operator" in self.options.keys() and self.options['operator'].upper() == "OR":
            inc_ind = np.logical_or(self.tests[0].include(), self.tests[1].include())
        elif "operator" in self.options.keys() and self.options['operator'].upper() == "XOR":
            inc_ind = np.logical_xor(self.tests[0].include(), self.tests[1].include())
        else:
            inc_ind = np.logical_and(self.tests[0].include(), self.tests[1].include())

        return inc_ind

class fixed_income_filter_numeric(fixed_income_filter):
    ''' Filter for including items that meet numeric requirement '''

    def include(self):
        """ produces series of booleans indicating inclusions """
        if 'exclude' in self.options.keys() and self.options['exclude'] and\
                self.options['exclude'] in self.df.columns:
            inc_ind = np.logical_not(self.df[self.options["exclude"]])
        else:
            inc_ind = pd.Series(np.ones(self.df.shape[0]) > 0, self.df.index)

        if str(self.options["func"]).upper() == 'GT':
            ind = self.df[self.options["key"]] > float(self.options["value"])
            inc_ind = np.logical_and(inc_ind, ind)
        elif str(self.options["func"]).upper() == 'GE':
            ind = self.df[self.options["key"]] >= float(self.options["value"])
            inc_ind = np.logical_and(inc_ind, ind)
        elif str(self.options["func"]).upper() == 'EQ':
            ind = self.df[self.options["key"]] == float(self.options['value'])
            inc_ind = np.logical_and(inc_ind, ind)
        elif str(self.options["func"]).upper() == 'LE':
            ind = self.df[self.options["key"]] <= float(self.options['value'])
            inc_ind = np.logical_and(inc_ind, ind)
        elif str(self.options["func"]).upper() == 'LE':
            ind = self.df[self.options["key"]] < float(self.options['value'])
            inc_ind = np.logical_and(inc_ind, ind)
        elif str(self.options["func"]).upper() == 'TOL' and 'tolerance' in self.options.keys():
            ind = np.abs(self.df[self.options["key"]] - float(self.options['value'])) <\
                float(self.options['tolerance'])
            inc_ind = np.logical_and(inc_ind, ind)
        else:
            raise ValueError("Faulty function definition")
        return inc_ind

class repeated_maturity_filter(fixed_income_filter):
    """ Filter for constructing UNIQUE maturity in portfoilio of FI securities """

    def __init__(self, options: Dict, df, dbg=False):
        """ Constructor for maturity filter """
        super().__init__(options, df, dbg=dbg)

        if self.options["name"] not in df.columns or\
                self.options["key"] not in df.columns:
            raise ValueError("name and key must columns in data frame")

        if "exclude" in self.options.keys() and self.options["exclude"] in df.columns:
            self.results = {
                "meta": {"ones": 0, "other": 0},
                "data": {},
                "exclusion": df[self.options["exclude"]].copy(),
            }
        else:
            self.results = {
                "meta": {"ones": 0, "other": 0},
                "data": {},
                "exclusion": pd.Series((np.zeroes(df.shape[0]) > 1), df.index),
            }
            if self.dbg:
                print("Warning -- excluded missing")


        for row in df.iterrows():
            val = round(row[1][self.options["key"]], self.options["value"])
            if val in self.results["data"].keys():
                choice = self.results["data"][val]["choice"]
                self.results["data"][val]["count"] += 1
                self.results["data"][val]["cusips"].append(row[0])
                self.results["data"][val]["choice"] = min2(
                    (choice, float(df.loc[choice, self.options["name"]])),
                    (row[0], float(row[1][self.options["name"]])),
                )

            else:
                self.results["data"][val] = {
                    "count": 1,
                    "cusips": [row[0]],
                    "choice": row[0],
                }

        for key, val in self.results["data"].items():
            if int(val["count"]) > 1:
                if self.dbg:
                    print(key, val)
                self.results["meta"]["other"] += 1
            else:
                self.results["meta"]["ones"] += 1

    def get_gt_one_count(self):
        """ produces the GT > 1 counts """
        return self.results["meta"]["other"]

    def include(self):
        """ Method for calculating the inclusion set """

        if self.get_gt_one_count() > 0:
            for _, val in self.results["data"].items():
                if val["count"] > 1:
                    fnl_set = set(val["cusips"]) - set([val["choice"]])
                    self.results['exclusion'][fnl_set] = True
        ind_inc = self.results['exclusion'].copy()
        ind_inc = np.logical_not(ind_inc)
        return ind_inc
