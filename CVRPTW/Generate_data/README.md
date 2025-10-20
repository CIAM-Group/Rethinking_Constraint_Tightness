We include the code of generating the CVRPTW instances with different TW tightness degrees, based on the code of [MVMoE](https://github.com/RoyalSkye/Routing-MVMoE).

To generate the data, run it:

```shell
bash generate_data.sh
```

An error may occur:

`subprocess.CalledProcessError: Command 'hgs/HGS-VRPTW/genvrp ....../0.hgs.tour -it 2000 -seed 1234' returned non-zero exit status 126.`

The root cause of this bug is a permission issue with the external executable file your Python script is trying to run.

Try this command to grant executable permissions to the genvrp file:
`chmod +x hgs/HGS-VRPTW/genvrp`

If it still does not work, please follow the code of [MVMoE](https://github.com/RoyalSkye/Routing-MVMoE) to reinstall hgs. 
