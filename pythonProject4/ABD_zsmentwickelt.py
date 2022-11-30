import numpy as np

# Regel: Linke Schichten entsprechen unteren Schichten, rechte oberen.

print("-------------------------------------")
print("All rights reserved by LinShi at TU Dresden")

# lokale Steifigkeitsmatrix

print("MSV_Analyse_A_Matrixberechnung")
print("--------------------------------------------")
print("1.Teil: Matrix der reduzierten Steifigkeit Q")
print("Geben Sie die Werkstoffkennwerte ein: ")
print()

E1 = float(input("Bitte geben Sie E1 in MPa ein: "))
E2 = float(input("Bitte geben Sie E2 in Mpa ein: "))
v12 = float(input("Bitte geben Sie v12 ein: "))
G12 = float(input("Bitte geben Sie G12 in MPa ein: "))
v21 = v12 * E2 / E1
Qij = np.array([[E1 / (1 - v12 * v21), v12 * E2 / (1 - v12 * v21), 0],
                [v12 * E2 / (1 - v12 * v21), E2 / (1 - v12 * v21), 0], [0, 0, G12]])
R_Qij = np.around(Qij, decimals=1)
print("Matrix der reduzierten Steifigkeit Q ist: ")
print(R_Qij)
print("Einheit: N/mm**2")
print()

print("----------------------------------------------")
print("2.Teil: Berechnung der Steifigkeitsmatrix einzelner Schichten")
print()

# Berechnung der ABD Matrix

n = int(input("Wie viele Schichten gibt es? : "))
h = float(input("Gesamtdicke in mm? : "))
hk = h / n
z0 = -h / 2
A = np.zeros((3, 3))
B = np.zeros((3, 3))
D = np.zeros((3, 3))
t = []
for i in np.arange(1, n + 1):
    t.append(np.float(input("Der Winkel für den %i. Schicht in Grad ist: (von unten nach oben)" % i)))
    tr = t[i - 1] * np.pi / 180
    Ti = np.array([[(np.cos(tr)) ** 2, (np.sin(tr)) ** 2, -2 * np.cos(tr) * np.sin(tr)],
                   [(np.sin(tr)) ** 2, (np.cos(tr)) ** 2, 2 * np.cos(tr) * np.sin(tr)],
                   [np.cos(tr) * np.sin(tr), -np.cos(tr) * np.sin(tr), (np.cos(tr)) ** 2 - (np.sin(tr)) ** 2]])
    Qij_ = np.dot(np.dot(Ti, Qij), Ti.T)
    # print(np.around(Qij_,decimals=1))
    zi = i * hk + z0
    z_i = 0.5 * (zi + zi - hk)
    A += Qij_ * hk
    B += Qij_ * z_i * hk
    D += Qij_ * (z_i ** 2 + 1 / 12 * hk ** 2) * hk
print("----------------------------------------------")
print()
print("Die A Matrix für das Laminat ist: ")
R_A = np.around(A, decimals=1)
print(R_A)
print("Einheit: N/mm")
print()
print("Die B Matrix für das Laminat ist: ")
R_B = np.around(B, decimals=1)
print(R_B)
print("Einheit: N")
print()
print("Die D Matrix für das Laminat ist: ")
R_D = np.around(D, decimals=1)
print(R_D)
print("Einheit: N*mm")
print()

ABD_Matrix = np.vstack((np.hstack((A, B)), np.hstack((B, D))))

# Berechnung der Ingenieurkonstanten

print("Die Ingenieurkonstanten Ex Ey Gxy vxy vyx sind: ")
Ex_A = 1 / np.linalg.inv(ABD_Matrix)[0, 0] / h
Ey_A = 1 / np.linalg.inv(ABD_Matrix)[1, 1] / h
Gxy_A = 1 / np.linalg.inv(ABD_Matrix)[2, 2] / h
vxy_A = -np.linalg.inv(ABD_Matrix)[1, 0] / np.linalg.inv(ABD_Matrix)[0, 0]
vyx_A = -np.linalg.inv(ABD_Matrix)[0, 1] / np.linalg.inv(ABD_Matrix)[1, 1]
print("Bitte beachten Sie: für die Berechnung der Ingenieurkonstanten wird nur die Scheibentheorie genutzt!")
print("Ex: ", np.around(Ex_A, decimals=1), ", Ey: ", np.around(Ey_A, decimals=1),
      ", Gxy: ", np.around(Gxy_A, decimals=1), ", vxy: ", np.around(vxy_A, decimals=1),
      ", vyx: ", np.around(vyx_A, decimals=1))
print()

# Berechnung der Verzerrungen
print("----------------------------------------------")
print("4.Teil: Berechnung der Laminatverzerrungen")
print()
Nx = float(input("Bitte geben Sie Schnittkraft Nx in N/mm ein: "))
Ny = float(input("Bitte geben Sie Schnittkraft Ny in N/mm ein: "))
Nxy = float(input("Bitte geben Sie Schnittkraft Nxy in N/mm ein "))
Mx = float(input("Bitte geben Sie Schnittmoment Mx in N ein: "))
My = float(input("Bitte geben Sie Schnittmoment My in N ein: "))
Mxy = float(input("Bitte geben Sie Schnittmoment Mxy in N ein: "))
N = np.array([[Nx], [Ny], [Nxy]])
M = np.array([[Mx], [My], [Mxy]])
Sch_G = np.vstack((N, M))
VzG = np.dot(np.linalg.inv(ABD_Matrix), Sch_G)
print("Die Verzerrungsgrößen sind: ")
print(np.around(VzG, decimals=4))
epsilon = np.array(VzG[0:3])
kappa = np.array(VzG[3:6])

# Spannungsanalyse
for i in range(1, n + 1):
    tr = t[i - 1] * np.pi / 180
    Ti = np.array([[(np.cos(tr)) ** 2, (np.sin(tr)) ** 2, -2 * np.cos(tr) * np.sin(tr)],
                   [(np.sin(tr)) ** 2, (np.cos(tr)) ** 2, 2 * np.cos(tr) * np.sin(tr)],
                   [np.cos(tr) * np.sin(tr), -np.cos(tr) * np.sin(tr), (np.cos(tr)) ** 2 - (np.sin(tr)) ** 2]])
    Qij_ = np.dot(np.dot(Ti, Qij), Ti.T)
    zo_i = z0 + i * hk
    zu_i = zo_i - hk
    sigma_o_glo = np.dot(Qij_, (epsilon + zo_i * kappa))
    sigma_o_lok = np.dot(np.linalg.inv(Ti), sigma_o_glo)
    print("Die lokalen Spannungen der oberen Seite in %i. Schicht in MPa sind: " % i)
    print(np.around(sigma_o_lok, decimals=1))
    # sigma_u_glo = np.dot(Qij_, (epsilon + zu_i * kappa))
    # sigma_u_lok = np.dot(np.linalg.inv(Ti), sigma_u_glo)
    # print("Die lokalen Spannung der unteren Seite in %i. Schicht in MPa sind: " % i)
    # print(np.around(sigma_u_lok, decimals=1))

print("-------------------------------------")
print("Congratulations boy! ! ! :)")
print()
