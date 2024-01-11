#pragma once

#include "marchee.cuh"

#include "S.cuh"

#define             C 13
#define         MAX_Y 1024
#define       BLOQUES 128
#define F_PAR_BLOQUES 8

#define      INSTS 4

#define FILTRES_PRIXS 0
#define         DOT1D 1
#define        LSTM1D 2
#define     LSTM1D_B8 3

// inst0 - "filtre_prixs"	// defaut. 0 pour bcp de fonctions 
// inst1 - "dot1d"
// inst2 - "lstm1d"

// mdl.quelconque  [        inst0            ][     inst1     ][ inst2 ][                   inst3                ]  T fois a la suite
//   ESPACE = somme( inst[i] )
// inst[i] :
//    [                   inst[i]                   ]
//    [           AUTRES           ][    SORTIES    ]
//
//	apres chaque optimisation les espaces doivent etre reinitialisés.
//		filtre a par exemple besoin de normaliser (mais on va pas le faire, juste pour voire)

typedef struct {
	//	Instructions
	uint   lignes[BLOQUES], decales[BLOQUES];
	uint * lignes__d, * decales__d;
	//
	uint insts[C];
	uint     Y[C];
	//
	uint inst_POIDS  [C];
	uint inst_VARS   [C];
	uint inst_LOCDS  [C];	//	infos de f(x) a ne pas re-calculer pendant df(x)
	uint inst_SORTIES[C];
	uint inst_DEPART_SORTIE[C];

	//
	float *  p[C];
	float *  y[C];
	float *  l[C];
	float * dy[C];
	float * dp[C];

	//
	float *  p__d[C];
	float *  y__d[C];
	float *  l__d[C];
	float * dy__d[C];
	float * dp__d[C];
} Mdl_t;

//	Memoire ram & vram
typedef void (*mdl_inst_f)(Mdl_t * mdl, uint inst);
extern mdl_inst_f cree_inst[INSTS];
//
Mdl_t * cree_mdl(
	uint Y[C],
	uint inst[C],
	uint lignes[C],
	uint decales[C]
);
//
void        liberer_mdl(Mdl_t * mdl);
//
void          mdl_verif(Mdl_t * mdl);
void   mdl_gpu_vers_cpu(Mdl_t * mdl);
void   mdl_cpu_vers_gpu(Mdl_t * mdl);
//
void       mdl_zero_cpu(Mdl_t * mdl);
void       mdl_zero_gpu(Mdl_t * mdl);
//
void mdl_zero_deriv_cpu(Mdl_t * mdl);
void mdl_zero_deriv_gpu(Mdl_t * mdl);
//
void  mdl_norme_filtres(Mdl_t * mdl);

//	I/O
Mdl_t * ouvrire_mdl(char * fichier             );
void     ecrire_mdl(Mdl_t * mdl, char * fichier);

//	Plume
extern char * nom_inst[INSTS];
extern mdl_inst_f plume_inst[INSTS];
void   plumer_mdl(Mdl_t * mdl                  );
void comportement(Mdl_t * mdl, uint t0, uint t1);
void    cmp_dy_dp(Mdl_t * mdl, uint t0, uint t1);
//
void mdl_plume_poids(Mdl_t * mdl);
//
void mdl_plume_grad(Mdl_t * mdl, uint t0, uint t1);
//
float    mdl_moy_dp(Mdl_t * mdl, uint c);

//	F & F'
typedef void (*mdl_f_f)(Mdl_t * mdl, uint inst, uint mode, uint t0, uint t1);
extern mdl_f_f inst_f [INSTS];
extern mdl_f_f inst_df[INSTS];
//
void  mdl_f(Mdl_t * mdl, uint t0, uint t1, uint mode);
void mdl_df(Mdl_t * mdl, uint t0, uint t1, uint mode);

//	Utilisation
float  mdl_score(Mdl_t * mdl, uint t0, uint t1, uint mode);
float * mdl_pred(Mdl_t * mdl, uint t0, uint t1, uint mode);
float   mdl_gain(Mdl_t * mdl, uint t0, uint t1, uint mode);
//
void mdl_aller_retour(Mdl_t * mdl, uint t0, uint t1, uint mode);