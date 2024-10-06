//
// Created by Shah Rukh Qasim on 17.07.2024.
//

#ifndef MY_PROJECT_SLIMFILMSENSITIVEDETECTOR_HH
#define MY_PROJECT_SLIMFILMSENSITIVEDETECTOR_HH

#include "G4VSensitiveDetector.hh"
#include "G4Step.hh"
#include "G4HCofThisEvent.hh"
#include "G4TouchableHistory.hh"
#include "G4SDManager.hh"
#include "G4Track.hh"
#include "G4StepPoint.hh"
#include "G4ThreeVector.hh"
#include "G4ParticleDefinition.hh"
#include "G4VProcess.hh"
#include <unordered_map>



class SlimFilmSensitiveDetector : public G4VSensitiveDetector {
public:
    SlimFilmSensitiveDetector(const G4String& name, const bool& storeFull);
    virtual ~SlimFilmSensitiveDetector();

    virtual void Initialize(G4HCofThisEvent* hce) override;
    virtual G4bool ProcessHits(G4Step* aStep, G4TouchableHistory* ROhist) override;
    virtual void EndOfEvent(G4HCofThisEvent* hce) override;
    void addSensitiveLayerInfo(int instanceId, int layerNo);

protected:
    std::unordered_map<int, int> instanceIdToLayerNo;

public:
    std::vector<double> chargeDeposit;
//    std::vector<int> layer;
    int numLayers;

    std::vector<double> depositFullCharge;
    std::vector<double> depositFullX;
    std::vector<double> depositFullY;
    std::vector<double> depositFullZ;
    bool storeFull;
};


#endif //MY_PROJECT_SLIMFILMSENSITIVEDETECTOR_HH
