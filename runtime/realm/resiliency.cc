//sri


#include "realm/resiliency.h"

namespace Realm {
	
	ResiliencyRequestTechnique::ResiliencyRequestTechnique(){
	}
	
	ResiliencyRequestTechnique::~ResiliencyRequestTechnique(){
	}

	template<typename T>
	ResiliencyRequestTechnique& ResiliencyRequestTechnique::add_resiliency() {
		//auto ret = requested_resilience_techniques.emplace(T());
		//if(!ret.second) 
		requested_resilience_techniques.insert((ResiliencyID)T::ID);
		return *this;
	}
	
	
	ResiliencyRequestTechnique& ResiliencyRequestTechnique::add_resiliency(ResiliencyID rid) {
		requested_resilience_techniques.insert(rid);
		return *this;
	}
	
	ResiliencyRequestTechnique& ResiliencyRequestTechnique::add_technique(Processor response_proc, 
		Processor::TaskFuncID response_task_id, const void *payload, size_t payload_size) {
		assert(0);
	}
	
	bool ResiliencyRequestTechnique::wants_technique(ResiliencyID technique) {
		if(requested_resilience_techniques.find(technique) != requested_resilience_techniques.end())
		return true;
		else
		return false;
	}
	
	size_t ResiliencyRequestTechnique::request_count() const {
		return (requested_resilience_techniques.size());
	}
	
	bool ResiliencyRequestTechnique::empty() const{
		return (requested_resilience_techniques.empty());
	}
	
	void ResiliencyRequestTechnique::clear() {
		requested_resilience_techniques.clear();
	}
	
};