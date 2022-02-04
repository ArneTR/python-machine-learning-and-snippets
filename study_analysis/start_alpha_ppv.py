import pandas as pd; pd.set_option('mode.chained_assignment','raise');
import numpy as np
import scipy
from scipy.stats.morestats import Std_dev

############ DEFINITONS

# Alpha = Fehler erster Art. False Positive. Wir behaupten einen Effekt, obwohl keiner da ist. (typischerweise = 5%)

# 1-Alpha = WK für: Wir behaupten keinen Effekt, und es ist auch keiner da (H0, dass beide Mittelwerte gleich sind, ist wahr). (True Positive)

# Beta = Fehler zweiter Art. False Negative. Wir behaupten KEINEN Effekt, obwohl es einen gibt.

#Power: 1-beta
# Diese ist die WK einen Effekt zu finden, sofern es einen gibt. (True Negative) -> Man nennt es True negative, da man H0 ablehnt ... die These, dass beide Mittelwerte gleich sind.


# Allgemein gilt:
# - Je größer n, desto kleiner Alpha und Beta
# - Je größer Alpha, desto kleiner beta und umgekehrt.
# - Je größer der Effekt, desto kleiner Alpha und Beta
###############################################################################################################


## first we talk about some easiy calculations we do BEFORE looking at the gathered data.
# Here we usually check assumptions and modelling and if they are valid.
# This includes power calculation, sample size calculation etc.

## Here we to calculate a type_I error based on a ALREADY set cutoff value  (assumption is we are dealing with normal distribution. No samples)
# Sometimes cutoff values are known before. But this is realyy seldom, so this calculation is most often not used

# If the cholesterol level of healthy men is normally distributed with a mean of 180 and a standard deviation of 20,
#  and men with cholesterol levels over 225 are diagnosed as not healthy, what is the probability of a type one error?
population_mean = 180
population_std_dev = 20
measured_value = 225
1-scipy.stats.norm.cdf((measured_value-population_mean) / population_std_dev) # probability_of_type_I_error

## Here we calculate the cutoff value based on a WANTED type I error (assumption is we are dealing with normal distribution. No samples)
# This is usually done to help with a wanted effect size assumption. For power-testing we NEED an estimate / wish for the effect.

#If the cholesterol level of healthy men is normally distributed with a mean of 180 and a standard deviation of 20, 
# at what level (in excess of 180) should men be diagnosed as not healthy if you want the probability of a type one error to be 0.02?
wanted_type_I_error = 0.02
scipy.stats.norm.ppf(1-wanted_type_I_error) * population_std_dev + population_mean # cutoff value for results, that are considered significant




######### DEFINTION of Alpha ##############
# If there were actually no effect (if the true difference between means were zero) then the probability 
# of observing a value for the difference equal to, or greater than, that actually observed would be p=0.05. I
# n other words there is a 5% chance of seeing a difference at least as big as we have done, by chance alone.

# This is VERY different from saying: There is only a 5% chance, that the result is wrong. NO. It is a 5% chance
# that GIVEN that there is no effect we randomly STILL GET a positive result.
# This is not enough to conclude about the chance of having spotted an acutal effect. You need beta and the probability
# of the hypothesis being TRUE. (as alpha is P(Positive result by random chance| H0 = True))


#### PPV Value
## This value is the likelihood of having a REAL result given the current positive result ratio (R) in the field (prior probability)
## R is heavily influenced by other studies on the same subject and also by prior studies from the same team. For instance
## If many mouse-models and also many low-n studies have shown that there might be an effect, R may be estimated higher
## Note R is ALWAYS an estimate and cannot be derived. It should however be estimated as a lower-boundary 

R = 0.4 # Ratio of Results to Non-Results. Example: Usually every third study in the field finds a PROVEN result. Then: R = 1/3
β = 0.2 # Important. THis is beta. Not Power!
u = 0.2 # Bias. This is factored in, cause every study has bias, through errors, bad randomization etc. 0.05 is the lowest value. 0.8 is very high and seen as a "worst-case".
α = 0.05
PPV = ((1 - β)*R + u*β*R) / (R + α - β*R + u - u*α + u*β*R)
PPV

# Was mich halt richtig stört: Es gibt widersprüchliche Aussage zur Power Calculation. Einmal geht es gar nicht für Hyptohesen wo H1 > y ist. 
# Dann ist es in anderen Rechnern wieder kein PRoblem. Oft gibt es viele Rechnenwege für das gleiche.
# SUPER nervig.
# Dann auch auch diese arbiträren Festlegungen: Cohen sagt, dass die Power von 0.8 ein hoher Wert ist. TOLL! Vielen Dank!
#
# Dann auch diese Annahmen die immer getroffen sein müssen: Muss eine Zufallsgezogene Stichprobe sein, der Mittelwert muss geschätzt werden, die Varianz muss geschätzt werden
# usw. ... iwo macht man immer Fehler ... und selbst wenn nicht sind die Rechnenvorschriften nur "Vorschläge", keine logischen beweisbaren Sachen. KOTZ!
#
# In dem Artikel https://royalsocietypublishing.org/doi/10.1098/rsos.140216#d3e749 sieht man schön, dass man das Konzept beliebig verbiegen kann.
# WEnn man den p-Wert von verschiendenen Standpunkten betrachtet oder das Experiment wiederholt, dann kommt nur Murks raus.
# 
# Schlussendlich bleiben einfache Gesetze: Man wiederhole eine Experiment 5 mal und gucke wie oft das gleiche Ergebnis rauskommt. Fertig.
# Der Wahnsinn aus einem Experiment eine Aussage treffen zu können und dann oft auch noch das Minimum an Probanden zu nehmen ist einfach mathematisch kaum zu halten.
# Ein einfacher Test mit p=0.05 hat lediglich die Aussagekraft: Kann man sich mal näher angucken. Aber bei weitem nicht mehr.

# WEiterhin sind folgende Regeln gut, falls es nicht möglich ist eine Studie zu wiederholen: Sample Size Calculations + p=0.001
 

