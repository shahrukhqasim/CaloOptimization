//
// Created by Shah Rukh Qasim on 17.07.2024.
//

#include "SlimFilmSensitiveDetector.hh"
#include "G4UnitsTable.hh"
#include "G4SystemOfUnits.hh"



SlimFilmSensitiveDetector::SlimFilmSensitiveDetector(const G4String &name, const bool& storeFull) : G4VSensitiveDetector(name) {
    numLayers = 0;
    SlimFilmSensitiveDetector::storeFull = storeFull;


}

SlimFilmSensitiveDetector::~SlimFilmSensitiveDetector() {}

void SlimFilmSensitiveDetector::Initialize(G4HCofThisEvent *hce) {
    std::fill(chargeDeposit.begin(), chargeDeposit.end(), 0);
//    layer.clear();
    if (storeFull) {
        depositFullX.clear();
        depositFullY.clear();
        depositFullZ.clear();
        depositFullCharge.clear();
    }
}

G4bool SlimFilmSensitiveDetector::ProcessHits(G4Step *aStep, G4TouchableHistory *ROhist) {
    auto theTrack = aStep->GetTrack();
    auto momentum = theTrack->GetMomentum();
    int instanceId = aStep->GetPreStepPoint()->GetPhysicalVolume()->GetInstanceID();

    int pdgId = theTrack->GetParticleDefinition()->GetPDGEncoding();

    if (pdgId==13 or pdgId==-13) {
        G4double momentumInGeV = momentum.mag() / GeV;

        if (momentumInGeV > 10) {
            std::cout << "Saw a muon... "<<momentumInGeV<<std::endl;
            theTrack->SetTrackStatus(fStopAndKill);
        }
        else {
//            std::cout << "Saw a low energy muon... "<<momentumInGeV<<std::endl;
        }
    }

    double dep = aStep->GetTotalEnergyDeposit() / GeV;
    if (dep>0){
//        chargeDeposit.push_back(dep);
        chargeDeposit[instanceIdToLayerNo[instanceId]] = chargeDeposit[instanceIdToLayerNo[instanceId]] + dep;
    }

    if (dep>0) {
        if (storeFull) {
            auto preStepPoint = aStep->GetPreStepPoint();
            auto position = preStepPoint->GetPosition();
            depositFullX.push_back(position.getX());
            depositFullY.push_back(position.getY());
            depositFullZ.push_back(position.getZ());
            depositFullCharge.push_back(dep);
        } else {
        }
    }

    return true;
}

void SlimFilmSensitiveDetector::EndOfEvent(G4HCofThisEvent *hce) {


}

void SlimFilmSensitiveDetector::addSensitiveLayerInfo(int instanceId, int layerNo) {
    instanceIdToLayerNo[instanceId] = layerNo;
    numLayers = std::max(numLayers, layerNo+1);
    chargeDeposit.push_back(0.0);
}