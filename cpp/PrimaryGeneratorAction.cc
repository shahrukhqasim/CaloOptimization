#include "PrimaryGeneratorAction.hh"
#include "G4ParticleGun.hh"
#include "G4ParticleTable.hh"
#include "G4ParticleDefinition.hh"
#include "G4SystemOfUnits.hh"
#include <iostream>

PrimaryGeneratorAction::PrimaryGeneratorAction()
: G4VUserPrimaryGeneratorAction()
{
}

PrimaryGeneratorAction::~PrimaryGeneratorAction()
{
    delete fParticleGun;
}

void PrimaryGeneratorAction::GeneratePrimaries(G4Event* anEvent)
{
    if(m_steppingAction!=NULL) {
        m_steppingAction->num_steps = 0;
    }
    else {
        std::cout<<"Problem!"<<std::endl;
    }
    G4ThreeVector position(next_x*m, next_y*m, next_z*m);
    G4ThreeVector momentum(next_px*GeV, next_py*GeV, next_pz*GeV);
    G4double time = 0;
    G4ParticleTable* particleTable = G4ParticleTable::GetParticleTable();
    G4ParticleDefinition* particleDefinition = particleTable->FindParticle(next_pdgid);

    if ( ! particleDefinition ) {
    G4cerr << "Error: PDGID " << next_pdgid << " not found in G4ParticleTable" << G4endl;
    exit(1);
    }
    // Create primary particle
    G4PrimaryParticle* primaryParticle = new G4PrimaryParticle(particleDefinition);
    primaryParticle->SetMomentum(momentum.x(), momentum.y(), momentum.z());
    primaryParticle->SetMass(particleDefinition->GetPDGMass());
    primaryParticle->SetCharge( particleDefinition->GetPDGCharge());

    std::cout<<"MMM: "<<primaryParticle->GetTotalMomentum() / GeV<<std::endl;

    // Create vertex
    G4PrimaryVertex* vertex = new G4PrimaryVertex(position, time);
    vertex->SetPrimary(primaryParticle);
    anEvent->AddPrimaryVertex(vertex);
}

void PrimaryGeneratorAction::setSteppingAction(CustomSteppingAction* steppingAction) {
    m_steppingAction = steppingAction;
}


void PrimaryGeneratorAction::setNextMomenta(double nextPx, double nextPy, double nextPz) {
    next_px = nextPx;
    next_py = nextPy;
    next_pz = nextPz;
}


void PrimaryGeneratorAction::setNextPosition(double nextX, double nextY, double nextZ) {
    next_x = nextX;
    next_y = nextY;
    next_z = nextZ;
}

void PrimaryGeneratorAction::setNextPdgId(int pdgid) {
    PrimaryGeneratorAction::next_pdgid = pdgid;
}
