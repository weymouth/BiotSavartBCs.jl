using KernelAbstractions
using KernelAbstractions: get_backend,@kernel,@index,@Const
KernelAbstractions.get_backend(nt::NTuple) = get_backend(first(nt))

# Loop macro over an index vector R
macro vecloop(args...)
    ex,_,itr = args
    _,I,R = itr.args; sym = []
    # grab arguments and replace composites
    WaterLily.grab!(sym,ex)
    setdiff!(sym,[I]) # don't want to pass index as an argument
    @gensym kern ind  # generate unique names
    return quote
        @kernel function $kern($(WaterLily.rep.(sym)...)) # replace composite arguments
            $ind = @index(Global,Linear) # linear index
            @inbounds $I = $R[$ind]      # this is expensive unless R is a vector
            @fastmath @inbounds $ex
        end
        $kern(get_backend($(sym[1])),64)($(sym...),ndrange=length($R))
    end |> esc
end
