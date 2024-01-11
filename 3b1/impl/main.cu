#include "main.cuh"

#include "../impl_tmpl/tmpl_etc.cu"

static void plume_pred(Mdl_t * mdl, uint t0, uint t1) {
	float * ancien = mdl_pred(mdl, t0, t1, 3);
	printf("PRED GENERALE = ");
	FOR(0, p, P) printf(" %f%% ", 100*ancien[p]);
	printf("\n");
	free(ancien);
};

float pourcent_masque_nulle[C] = {0};

float * pourcent_masque = de_a(0.80, 0.0, C);

//	! A FAIRE ! :
//		selection (mutation de +/- 1 ligne (de meme source))
//

float * alpha = de_a(1e-3, 1e-3, C);

PAS_OPTIMISER()
int main(int argc, char ** argv) {
	MSG("S(x) Ajouter un peut d'aléatoire");
	MSG("S(x) Eventuellement faire des prediction plus lointaines");
	//	-- Init --
	srand(0);
	cudaSetDevice(0);
	titre(" Charger tout ");   charger_tout();

	//	-- Verification --
	//titre("Verifier MDL");     verif_mdl_1e5();

	//===============
	titre("  Programme Generale  ");

	uint Y[C] = {
		1024,
		512,512,
		256,256,
		128,128,
		64,
		32,
		16,
		8,
		4,
		P
	};
	uint insts[C] = {
		FILTRES_PRIXS,
		DOT1D,DOT1D,
		DOT1D,DOT1D,
		DOT1D,DOT1D,
		DOT1D,
		DOT1D,
		DOT1D,
		DOT1D,
		DOT1D,
		DOT1D
	};
	//
	uint lignes[BLOQUES] = {0};
	FOR(0, i, BLOQUES) lignes[i] = rand() % EMA_INTS;
	//
	uint decales[BLOQUES] = {0};
	FOR(0, i, BLOQUES) decales[i] = rand() % MAX_DECALES;
	//	Assurances :
	FOR(0, i, 3) {
		lignes [i] = 0;
		decales[i] = 0;
	}
	//
	Mdl_t * mdl = cree_mdl(Y, insts, lignes, decales);

	//Mdl_t * mdl = ouvrire_mdl("mdl.bin");

	plumer_mdl(mdl);

	//	================= Initialisation ==============
	uint t0 = DEPART;
	uint t1 = ROND_MODULO(FIN, (16*16));
	printf("t0=%i t1=%i FIN=%i (t1-t0=%i, %%(16*16)=%i)\n", t0, t1, FIN, t1-t0, (t1-t0)%(16*16));
	//
	plume_pred(mdl, t0, t1);
	//
	uint REP = 150;
	FOR(0, rep, REP) {
		FOR(0, i, 5) {
			printf(" ================== %i/20 ================\n", i);
			optimisation_mini_packet(
				mdl,
				t0, t1, 16*16*1,
				alpha, 1.0,
				RMSPROP, 70,
				pourcent_masque);
			plume_pred(mdl, t0, t1);
			mdl_gpu_vers_cpu(mdl);
			ecrire_mdl(mdl, "mdl.bin");
		}
		//
		optimiser(
			mdl,
			t0, t1,
			alpha, 1.0,
			RMSPROP, 1000,
			pourcent_masque_nulle);
		//
		mdl_gpu_vers_cpu(mdl);
		ecrire_mdl(mdl, "mdl.bin");
		plume_pred(mdl, t0, t1);
		printf("===================================================\n");
		printf("==================TERMINE %i/%i=======================\n", rep+1, REP);
		printf("===================================================\n");
	}
	//
	mdl_gpu_vers_cpu(mdl);
	ecrire_mdl(mdl, "mdl.bin");
	liberer_mdl(mdl);

	//	-- Fin --
	liberer_tout();
};