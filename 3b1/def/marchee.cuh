#pragma once

#define DEBBUG false

#include "etc.cuh"

#define PRIXS 55548
//#define L 1  		//u += u*f*levier*(p[i+L]/p[i]-1)
#define P 1 //2 ou 3

#define N_FLTR  8
#define N N_FLTR

#define MAX_INTERVALLE 256
#define MAX_DECALES 32

#define DEPART ((N_FLTR+MAX_DECALES)*MAX_INTERVALLE)
#if (DEBBUG == false)
	#define FIN (PRIXS-P-1)
#else
	#define FIN (DEPART+1)
#endif

typedef struct {
	uint     ligne;
	uint       ema;
	uint    interv;
	float * source;
} ema_int;

#define EMA_INTS (61)

/*	Note : dans `normalisee` et `dif_normalisee`
les intervalles sont deja calculee. Donc tout
ce qui est avant DEPART n'est pas initialisee.
*/

//	Sources
extern float   prixs[PRIXS];	//  prixs.bin
extern float   macds[PRIXS];	//   macd.bin
extern float volumes[PRIXS];	// volume.bin
extern float   hight[PRIXS];	//  prixs.bin
extern float     low[PRIXS];	//  prixs.bin

//	ema des sources
extern float            ema[EMA_INTS * PRIXS];
extern float     normalisee[EMA_INTS * PRIXS * N_FLTR];
extern float dif_normalisee[EMA_INTS * PRIXS * N_FLTR];

//	======================================

//	Sources en GPU
extern float *   prixs__d;	//	nVidia
extern float *   macds__d;	//	nVidia
extern float * volumes__d;	//	nVidia
extern float *   hight__d;	//	nVidia
extern float *     low__d;	//	nVidia

//	gpu ema des sources
extern float *            ema__d;	//	nVidia
extern float *     normalisee__d;	//	nVidia
extern float * dif_normalisee__d;	//	nVidia

void      charger_les_prixs();
void calculer_ema_norm_diff();
void    charger_vram_nvidia();

void     liberer_cudamalloc();

static ema_int ema_ints[EMA_INTS] = {
//	 id   ema  interv  source
// --------- Close ---------
	{ 0,   1,   1,   prixs },
	{ 1,   2,   1,   prixs },
	{ 2,   5,   1,   prixs },
	{ 3,   4,   4,   prixs },
	{ 4,   8,   4,   prixs },
	{ 5,   20,   4,   prixs },
	{ 6,   16,   16,   prixs },
	{ 7,   32,   16,   prixs },
	{ 8,   80,   16,   prixs },
	{ 9,   64,   64,   prixs },
	{ 10,   128,   64,   prixs },
	{ 11,   320,   64,   prixs },
// --------- High ---------
	{ 12,   1,   1,   hight },
	{ 13,   2,   1,   hight },
	{ 14,   5,   1,   hight },
	{ 15,   4,   4,   hight },
	{ 16,   8,   4,   hight },
	{ 17,   20,   4,   hight },
	{ 18,   16,   16,   hight },
	{ 19,   32,   16,   hight },
	{ 20,   80,   16,   hight },
	{ 21,   64,   64,   hight },
	{ 22,   128,   64,   hight },
	{ 23,   320,   64,   hight },
// --------- Low ---------
	{ 24,   1,   1,   low },
	{ 25,   2,   1,   low },
	{ 26,   5,   1,   low },
	{ 27,   4,   4,   low },
	{ 28,   8,   4,   low },
	{ 29,   20,   4,   low },
	{ 30,   16,   16,   low },
	{ 31,   32,   16,   low },
	{ 32,   80,   16,   low },
	{ 33,   64,   64,   low },
	{ 34,   128,   64,   low },
	{ 35,   320,   64,   low },
// --------- Macd ---------
	{ 36,   1,   1,   macds },
	{ 37,   2,   1,   macds },
	{ 38,   5,   1,   macds },
	{ 39,   4,   4,   macds },
	{ 40,   8,   4,   macds },
	{ 41,   20,   4,   macds },
	{ 42,   16,   16,   macds },
	{ 43,   32,   16,   macds },
	{ 44,   80,   16,   macds },
	{ 45,   64,   64,   macds },
	{ 46,   128,   64,   macds },
	{ 47,   320,   64,   macds },
// --------- Volumes ---------
	{ 48,   1,   1,   volumes },
	{ 49,   2,   1,   volumes },
	{ 50,   5,   1,   volumes },
	{ 51,   4,   4,   volumes },
	{ 52,   8,   4,   volumes },
	{ 53,   20,   4,   volumes },
	{ 54,   16,   16,   volumes },
	{ 55,   32,   16,   volumes },
	{ 56,   80,   16,   volumes },
	{ 57,   64,   64,   volumes },
	{ 58,   128,   64,   volumes },
	{ 59,   320,   64,   volumes },
//  plus grand interv que ema
    { 60,    5,   15,   volumes }
};

void charger_tout();
void liberer_tout();