# Extending the function push! to ensure the Observable is triggered
#
# Observables are triggered when they get assigned a value,
# but when appending the list is mutated, not assigned, and therefore the Observable must be triggered manually
import Base.push!
function push!(list_obs::Observable{Vector{Float32}}, value::Float32)
    push!(list_obs[], value)
    list_obs[] = list_obs[]
end